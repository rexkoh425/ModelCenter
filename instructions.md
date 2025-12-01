Setup for models submodule

1) Initialize submodule
- From the repo root:
  git submodule update --init --recursive

2) Download required weights (examples)
- TinyLlama base (HF): TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Place safetensors under llm/tinyllama/ (or GGUF under llm/tinyllama/ for inference via llama.cpp).
- TinyLlama LoRA adapters: store under llm/tinyllama_finetuned_v*/ (kept gitignored).
- LaneATT checkpoints: place under vision/LaneATT/checkpoints/ or experiments/*/models/.
- YOLOP/YOLOPv2: download official weights into vision/YOLOP/weights/ and vision/YOLOPv2/data/weights/.
- Whisper binaries: place ggml/onnx under audio/whisper/.

3) Keep large files out of git
- models/.gitignore is prefilled for *.pt, *.bin, *.gguf, optimizer.pt, LaneATT zips, YOLOP/YOLOPv2 packed objects, TinyLlama splits, etc.

4) Suggested layout
- llm/tinyllama/      # base weights, tokenizer, configs
- llm/tinyllama_finetuned_v*/  # LoRA adapters/checkpoints (ignored)
- vision/LaneATT/     # lane detection models/checkpoints
- vision/YOLOP/, vision/YOLOPv2/  # lane/drive assist models
- audio/whisper/      # speech models (if used)

5) Performance tip
- Keep weights on SSD/NVMe; avoid network mounts for training/inference.

