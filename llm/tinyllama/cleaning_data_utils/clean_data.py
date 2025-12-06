#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple


ASSIST_TOK = "<|assistant|>"
USER_TOK = "<|user|>"
SYS_TOK = "<|system|>"

INTRO_PREFIXES = (
    "sure", "here", "use", "run", "try", "you can", "the command", "example", "note",
)

DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/\b",
    r"\bmkfs(\.|)\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-jsonl", required=True, help="Input JSONL with {query, command}")
    p.add_argument("--out-jsonl", required=True, help="Cleaned output JSONL")
    p.add_argument("--bad-jsonl", required=True, help="Rejected rows JSONL (with reason)")
    p.add_argument("--stats-json", required=True, help="Stats JSON output")

    p.add_argument("--require-one-line", action="store_true", help="Keep only first command line (recommended)")
    p.add_argument("--max-query-chars", type=int, default=400)
    p.add_argument("--max-cmd-chars", type=int, default=400)

    p.add_argument("--dedup", action="store_true", help="Deduplicate identical (query, command) pairs")
    p.add_argument("--drop-ambiguous", action="store_true", help="Drop queries that map to >1 distinct commands (recommended for exact-match)")

    p.add_argument("--safety-filter", action="store_true", help="Drop obviously dangerous commands")
    p.add_argument("--strip-chat-tokens", action="store_true", help="If command contains <|system|>/<|user|>/<|assistant|>, keep only assistant content")

    return p.parse_args()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_query(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def normalize_cmd(c: str) -> str:
    c = c.strip()
    c = c.replace("\r\n", "\n").replace("\r", "\n")
    c = re.sub(r"[ \t]+", " ", c)
    c = c.rstrip(";")
    return c.strip()


def strip_code_fences(text: str) -> str:
    s = text.strip()
    # If it contains ```...```, take the first fenced block content
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            block = parts[1]
            # remove optional language tag on first line
            block_lines = block.splitlines()
            if block_lines and re.match(r"^[a-zA-Z0-9_+-]+$", block_lines[0].strip()):
                block = "\n".join(block_lines[1:])
            return block.strip()
    return s


def extract_assistant_only(text: str) -> str:
    s = text
    if ASSIST_TOK in s:
        s = s.split(ASSIST_TOK, 1)[1]
    elif "Assistant:" in s:
        s = s.split("Assistant:", 1)[1]

    # Cut off any later role tokens
    for cut in (USER_TOK, SYS_TOK, ASSIST_TOK):
        if cut in s:
            s = s.split(cut, 1)[0]

    s = s.replace("<|eot_id|>", "").replace("</s>", "").replace("<|endoftext|>", "")
    return s.strip()


def choose_command_line(text: str) -> str:
    """
    Pick the first line that looks like a command and not explanatory text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Prefer lines that start with a typical command-y token (letter, /, ., ~)
    for ln in lines:
        low = ln.lower().strip()
        if any(low.startswith(p) for p in INTRO_PREFIXES):
            continue
        if re.match(r"^[A-Za-z0-9_./~$-]", ln):
            return ln

    # fallback: first non-empty line
    return lines[0]


def is_dangerous(cmd: str) -> Optional[str]:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, cmd):
            return pat
    return None


def canonicalize_shell(cmd: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse via shlex and re-join for a stable whitespace/quoting form.
    Returns (canonical_cmd, error_reason)
    """
    try:
        toks = shlex.split(cmd, posix=True)
    except Exception as e:
        return None, f"shlex_split_failed:{type(e).__name__}"
    if not toks:
        return None, "empty_after_split"
    try:
        canon = shlex.join(toks)
    except Exception as e:
        return None, f"shlex_join_failed:{type(e).__name__}"
    return canon, None


def main():
    args = parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    bad_path = Path(args.bad_jsonl)
    stats_path = Path(args.stats_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    seen_pairs = set()  # for dedup (hash of query||cmd)
    query_to_cmd = {}   # for ambiguity detection: query_hash -> cmd_hash

    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8") as fout, \
         bad_path.open("w", encoding="utf-8") as fbad:

        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            stats["seen_lines"] += 1

            try:
                obj = json.loads(line)
            except Exception:
                stats["reject_json_parse"] += 1
                fbad.write(json.dumps({"line": idx, "reason": "json_parse_failed", "raw": line}, ensure_ascii=False) + "\n")
                continue

            q = obj.get("query", "")
            c = obj.get("command", "")

            if not isinstance(q, str) or not isinstance(c, (str, dict, list, int, float, bool)) and c is not None:
                stats["reject_bad_types"] += 1
                fbad.write(json.dumps({"line": idx, "reason": "bad_types", "query": str(type(q)), "command": str(type(c))}, ensure_ascii=False) + "\n")
                continue

            q = normalize_query(str(q))
            c = str(c)

            if len(q) == 0:
                stats["reject_empty_query"] += 1
                fbad.write(json.dumps({"line": idx, "reason": "empty_query"}, ensure_ascii=False) + "\n")
                continue

            if len(q) > args.max_query_chars:
                stats["reject_query_too_long"] += 1
                fbad.write(json.dumps({"line": idx, "reason": "query_too_long", "query": q[:200]}, ensure_ascii=False) + "\n")
                continue

            # Clean command text
            c = strip_code_fences(c)
            if args.strip_chat_tokens:
                c = extract_assistant_only(c)
            if args.require_one_line:
                c = choose_command_line(c)

            c = normalize_cmd(c)

            if len(c) == 0:
                stats["reject_empty_command"] += 1
                fbad.write(json.dumps({"line": idx, "reason": "empty_command", "query": q}, ensure_ascii=False) + "\n")
                continue

            if len(c) > args.max_cmd_chars:
                stats["reject_cmd_too_long"] += 1
                fbad.write(json.dumps({"line": idx, "reason": "command_too_long", "query": q, "command": c[:200]}, ensure_ascii=False) + "\n")
                continue

            if args.safety_filter:
                pat = is_dangerous(c)
                if pat:
                    stats["reject_dangerous"] += 1
                    fbad.write(json.dumps({"line": idx, "reason": "dangerous_command", "pattern": pat, "query": q, "command": c}, ensure_ascii=False) + "\n")
                    continue

            # Canonicalize with shlex (stable whitespace/quoting)
            canon, err = canonicalize_shell(c)
            if err:
                stats["reject_unparseable_cmd"] += 1
                fbad.write(json.dumps({"line": idx, "reason": err, "query": q, "command": c}, ensure_ascii=False) + "\n")
                continue

            qn = q
            cn = canon

            # Dedup pairs
            if args.dedup:
                key = sha1(qn + "||" + cn)
                if key in seen_pairs:
                    stats["drop_duplicate_pair"] += 1
                    continue
                seen_pairs.add(key)

            # Ambiguity: same query maps to multiple commands
            qh = sha1(qn)
            ch = sha1(cn)
            if qh in query_to_cmd and query_to_cmd[qh] != ch:
                stats["ambiguous_query_hits"] += 1
                if args.drop_ambiguous:
                    stats["drop_ambiguous"] += 1
                    fbad.write(json.dumps(
                        {"line": idx, "reason": "ambiguous_query", "query": qn, "command": cn},
                        ensure_ascii=False
                    ) + "\n")
                    continue
            else:
                query_to_cmd[qh] = ch

            # Write cleaned row
            fout.write(json.dumps({"query": qn, "command": cn}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            if idx % 200000 == 0:
                print(f"[{idx}] kept={stats['kept']} rejected={stats['seen_lines']-stats['kept']}", file=sys.stderr)

    # Save stats
    stats_out = dict(stats)
    stats_out["kept_ratio"] = stats["kept"] / max(1, stats["seen_lines"])
    stats_out["ambiguous_query_rate"] = stats["ambiguous_query_hits"] / max(1, stats["seen_lines"])
    stats_out["unique_queries_seen"] = len(query_to_cmd)
    stats_out["unique_pairs_seen"] = len(seen_pairs) if args.dedup else None

    stats_path.write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    print("Done.")
    print(f"Input:  {in_path}")
    print(f"Kept:   {stats['kept']}")
    print(f"Output: {out_path}")
    print(f"Bad:    {bad_path}")
    print(f"Stats:  {stats_path}")


if __name__ == "__main__":
    main()
