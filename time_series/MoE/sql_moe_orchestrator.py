"""
Mixture-of-experts (MoE) orchestrator that drives Qwen3-8B to author SQL queries.

Example:
    python -m MoE.sql_moe_orchestrator --config MoE/sql_moe_orchestrator_config.yaml \\
        --query "Show weekly revenue per region for the last quarter."
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from Qwen3.qwen3_runner import Qwen3Runner, load_app_config


DEFAULT_CONFIG_FILENAME = "sql_moe_orchestrator_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
AVAILABLE_EXPERTS = ("planner", "sql_writer", "validator")


@dataclass(slots=True)
class DatabaseSettings:
    dialect: str
    schema_text: str


@dataclass(slots=True)
class RouterSettings:
    max_experts: int = 3
    temperature: float = 0.15
    top_p: float = 0.9


@dataclass(slots=True)
class RunSettings:
    user_query: Optional[str] = None
    extra_context: Optional[str] = None
    print_intermediate: bool = True


@dataclass(slots=True)
class SQLMoEConfig:
    qwen_runner_config_path: Path
    database: DatabaseSettings
    router: RouterSettings
    run: RunSettings


@dataclass(slots=True)
class RouterPlan:
    reasoning: str
    experts: List[str]


@dataclass(slots=True)
class SQLMoEResult:
    final_sql: str
    explanation: str
    router_reasoning: str
    expert_outputs: Dict[str, Dict[str, Any]]


@dataclass(slots=True)
class ExpertDefinition:
    name: str
    system_prompt: str
    template: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9


def _build_expert_definitions() -> Dict[str, ExpertDefinition]:
    return {
        "planner": ExpertDefinition(
            name="planner",
            system_prompt=(
                "You are a senior analytics engineer. Break business questions into "
                "clear SQL-friendly steps before any code is written."
            ),
            template=textwrap.dedent(
                """\
                User request:
                {user_query}

                Database dialect: {dialect}
                Declarative schema reference:
                {schema}

                Extra business context:
                {extra_context}

                Router summary:
                {router_reasoning}

                Previous expert outputs:
                {previous_outputs}

                Respond with compact JSON containing:
                  - "analysis": short natural language explanation.
                  - "selected_tables": array of table names you need.
                  - "joins": array explaining join paths.
                  - "filters": array describing WHERE clause ideas (with parameter hints).
                  - "aggregations": array for GROUP BY logic or metrics if relevant.
                  - "candidate_sql": optional draft SQL or CTE snippet.
                Keep JSON valid and do not add extra keys.
                """
            ),
            max_new_tokens=512,
            temperature=0.25,
            top_p=0.9,
        ),
        "sql_writer": ExpertDefinition(
            name="sql_writer",
            system_prompt=(
                "You are a SQL virtuoso. Produce runnable SQL that follows the supplied "
                "schema, dialect, and planner guidance. Prefer CTEs for clarity."
            ),
            template=textwrap.dedent(
                """\
                Goal: translate the request into executable SQL.

                Dialect: {dialect}
                Schema:
                {schema}

                Planner + router context:
                {router_reasoning}

                User request:
                {user_query}

                Extra context:
                {extra_context}

                Previous expert outputs:
                {previous_outputs}

                Return JSON with:
                  - "sql": final SQL string.
                  - "assumptions": array of clarifications you had to make.
                  - "notes": array describing tricky parts (e.g., window functions).
                SQL must be formatted with uppercase keywords and end with a semicolon.
                """
            ),
            max_new_tokens=512,
            temperature=0.15,
            top_p=0.85,
        ),
        "validator": ExpertDefinition(
            name="validator",
            system_prompt=(
                "You are a meticulous database QA analyst. Verify SQL matches the "
                "request, follows schema constraints, and is syntactically valid."
            ),
            template=textwrap.dedent(
                """\
                Review the candidate SQL and describe any issues. If fixes are required, update
                the SQL accordingly.

                Schema:
                {schema}

                Dialect: {dialect}
                Router reasoning:
                {router_reasoning}

                User request:
                {user_query}

                Extra context:
                {extra_context}

                Previous expert outputs:
                {previous_outputs}

                Return JSON with:
                  - "sql": approved (and possibly edited) SQL.
                  - "issues": array of strings describing problems found (empty if none).
                  - "tests": array of lightweight test ideas or query rewrites.
                  - "explanation": concise summary for the requester.
                """
            ),
            max_new_tokens=400,
            temperature=0.1,
            top_p=0.8,
        ),
    }


EXPERT_DEFINITIONS = _build_expert_definitions()


def load_sql_moe_config(config_path: Path) -> SQLMoEConfig:
    expanded = Path(config_path).expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"SQL MoE config not found: {expanded}")

    with expanded.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("SQL MoE config must be a mapping.")

    qwen_path_value = data.get("qwen_runner_config") or Path("Qwen3") / "qwen3_runner_config.yaml"
    qwen_config_path = Path(qwen_path_value)
    if not qwen_config_path.is_absolute():
        qwen_config_path = (expanded.parent / qwen_config_path).resolve()

    db_section = data.get("database", {})
    if not isinstance(db_section, dict):
        raise ValueError("Config section 'database' must be a mapping.")
    dialect = str(db_section.get("dialect", "postgresql"))
    schema_parts: List[str] = []
    schema_file = db_section.get("schema_file")
    if schema_file:
        schema_path = Path(schema_file)
        if not schema_path.is_absolute():
            schema_path = (expanded.parent / schema_path).resolve()
        if not schema_path.is_file():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        schema_parts.append(schema_path.read_text(encoding="utf-8").strip())
    inline_schema = db_section.get("inline_schema")
    if inline_schema:
        schema_parts.append(str(inline_schema).strip())
    schema_text = "\n\n".join(part for part in schema_parts if part).strip()
    if not schema_text:
        raise ValueError("Provide database.schema_file and/or database.inline_schema.")
    database = DatabaseSettings(dialect=dialect, schema_text=schema_text)

    router_section = data.get("router", {})
    if not isinstance(router_section, dict):
        raise ValueError("Config section 'router' must be a mapping.")
    max_experts = int(router_section.get("max_experts", 3))
    if max_experts < 1:
        max_experts = 1
    max_experts = min(max_experts, len(AVAILABLE_EXPERTS))
    router = RouterSettings(
        max_experts=max_experts,
        temperature=float(router_section.get("temperature", 0.15)),
        top_p=float(router_section.get("top_p", 0.9)),
    )

    run_section = data.get("run", {})
    if not isinstance(run_section, dict):
        raise ValueError("Config section 'run' must be a mapping.")
    run = RunSettings(
        user_query=(run_section.get("user_query") or None),
        extra_context=(run_section.get("extra_context") or None),
        print_intermediate=bool(run_section.get("print_intermediate", True)),
    )

    return SQLMoEConfig(
        qwen_runner_config_path=qwen_config_path,
        database=database,
        router=router,
        run=run,
    )


class SQLMoEOrchestrator:
    """Coordinates routing + experts to craft SQL using Qwen3-8B."""

    def __init__(self, config: SQLMoEConfig) -> None:
        self.config = config
        qwen_config, _ = load_app_config(self.config.qwen_runner_config_path)
        if qwen_config.download_only:
            raise ValueError("Qwen3 config must have download_only=False for inference.")
        self.runner = Qwen3Runner(qwen_config)

    def execute(
        self,
        user_query: str,
        extra_context: Optional[str] = None,
        *,
        verbose: Optional[bool] = None,
    ) -> SQLMoEResult:
        query = user_query.strip()
        if not query:
            raise ValueError("User query cannot be empty.")
        context = (extra_context or self.config.run.extra_context or "").strip()
        show_intermediate = self.config.run.print_intermediate if verbose is None else bool(verbose)

        router_plan = self._route(query, context)
        expert_history: Dict[str, Dict[str, Any]] = {}

        for expert_name in router_plan.experts:
            payload = self._run_expert(
                expert_name=expert_name,
                user_query=query,
                extra_context=context,
                router_plan=router_plan,
                history=expert_history,
            )
            expert_history[expert_name] = payload
            if show_intermediate:
                pretty = json.dumps(payload, indent=2, ensure_ascii=False)
                print(f"\n[{expert_name}]")
                print(pretty)

        final_sql = (
            expert_history.get("validator", {}).get("sql")
            or expert_history.get("sql_writer", {}).get("sql")
            or expert_history.get("planner", {}).get("candidate_sql")
            or ""
        )
        final_sql = final_sql.strip()
        if not final_sql:
            raise RuntimeError("Experts failed to produce SQL. Inspect intermediate outputs.")

        explanation = (
            expert_history.get("validator", {}).get("explanation")
            or expert_history.get("validator", {}).get("issues")
            or expert_history.get("sql_writer", {}).get("notes")
            or "SQL drafted by MoE orchestrator."
        )
        explanation_str = explanation if isinstance(explanation, str) else json.dumps(explanation, ensure_ascii=False)

        return SQLMoEResult(
            final_sql=final_sql,
            explanation=explanation_str,
            router_reasoning=router_plan.reasoning,
            expert_outputs=expert_history,
        )

    def _route(self, user_query: str, extra_context: str) -> RouterPlan:
        default_experts = list(AVAILABLE_EXPERTS[: self.config.router.max_experts])
        prompt = textwrap.dedent(
            f"""\
            You are the routing controller for a SQL mixture-of-experts assistant.
            Available experts (choose up to {self.config.router.max_experts}):
              - planner: define tables, joins, filters, metrics.
              - sql_writer: author executable SQL.
              - validator: audit correctness and adjust SQL if needed.

            Decide which experts should respond (ordered). Provide JSON with keys:
              - "reasoning": short text explaining your choices.
              - "experts": ordered array of expert names from {list(AVAILABLE_EXPERTS)}.

            User request: {user_query}
            Schema: {self.config.database.schema_text}
            Extra context: {extra_context or "N/A"}
            """
        )
        raw = self._generate(
            prompt=prompt,
            system_prompt=(
                "You route analysis and SQL generation tasks to specialist agents. "
                "Always emit strict JSON."
            ),
            max_new_tokens=256,
            temperature=self.config.router.temperature,
            top_p=self.config.router.top_p,
        )
        parsed = _safe_json_parse(raw)
        reasoning = str(parsed.get("reasoning") or raw).strip()
        experts = self._normalise_experts(parsed.get("experts"), default_experts)
        return RouterPlan(reasoning=reasoning, experts=experts)

    def _run_expert(
        self,
        expert_name: str,
        user_query: str,
        extra_context: str,
        router_plan: RouterPlan,
        history: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if expert_name not in EXPERT_DEFINITIONS:
            raise ValueError(f"Unknown expert requested: {expert_name}")
        definition = EXPERT_DEFINITIONS[expert_name]
        previous_outputs = _format_history(history)
        prompt = definition.template.format(
            user_query=user_query,
            schema=self.config.database.schema_text,
            dialect=self.config.database.dialect,
            extra_context=extra_context or "N/A",
            router_reasoning=router_plan.reasoning,
            previous_outputs=previous_outputs,
        )
        raw = self._generate(
            prompt=prompt,
            system_prompt=definition.system_prompt,
            max_new_tokens=definition.max_new_tokens,
            temperature=definition.temperature,
            top_p=definition.top_p,
        )
        return _safe_json_parse(raw)

    def _generate(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        return self.runner.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @staticmethod
    def _normalise_experts(candidates: Any, default: List[str]) -> List[str]:
        ordered: List[str] = []
        iterable: Iterable[str]
        if isinstance(candidates, list):
            iterable = candidates
        elif isinstance(candidates, str):
            iterable = [candidates]
        else:
            iterable = []

        for name in iterable:
            normalized = str(name).strip().lower()
            if normalized in AVAILABLE_EXPERTS and normalized not in ordered:
                ordered.append(normalized)
            if len(ordered) >= len(default):
                break

        if not ordered:
            ordered = list(default)
        else:
            ordered = ordered[: len(default)]
        return ordered


class SQLMoEOrchestratorApp:
    """Loads YAML config and executes MoE SQL generation flow."""

    def __init__(self, config_path: Path | None = None) -> None:
        resolved = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config_path = resolved
        self.config = load_sql_moe_config(self.config_path)
        self.orchestrator = SQLMoEOrchestrator(self.config)

    def run(
        self,
        user_query: Optional[str] = None,
        extra_context: Optional[str] = None,
        *,
        verbose: Optional[bool] = None,
    ) -> SQLMoEResult:
        query = (user_query or self.config.run.user_query or "").strip()
        if not query:
            query = input("Describe the SQL you need: ").strip()
        if not query:
            raise ValueError("A user query is required.")
        context = extra_context if extra_context is not None else self.config.run.extra_context
        return self.orchestrator.execute(query, context, verbose=verbose)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"text": candidate}


def _format_history(history: Dict[str, Dict[str, Any]]) -> str:
    if not history:
        return "No previous expert output yet."
    chunks: List[str] = []
    for name, payload in history.items():
        pretty = json.dumps(payload, ensure_ascii=False)
        chunks.append(f"{name}: {pretty}")
    return "\n".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3-based SQL MoE orchestrator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to sql_moe_orchestrator_config.yaml",
    )
    parser.add_argument("--query", type=str, help="Override run.user_query from the config.")
    parser.add_argument("--extra-context", type=str, help="Override run.extra_context from the config.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate expert dumps (only show final SQL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = SQLMoEOrchestratorApp(args.config)
    result = app.run(
        user_query=args.query,
        extra_context=args.extra_context,
        verbose=None if not args.quiet else False,
    )
    print("\n=== SQL MoE Result ===")
    print(f"Plan: {result.router_reasoning}")
    print("\nFinal SQL:")
    print(result.final_sql)
    print("\nExplanation:")
    print(result.explanation)


if __name__ == "__main__":
    main()
