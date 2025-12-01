Models Submodule

This directory is intended to be a separate git submodule that holds pretrained weights and task-specific models. Large binaries are gitignored (see models/.gitignore).

Contents (by area)
- llm/tinyllama: TinyLlama base weights, LoRA adapters, data splits.
- vision: LaneATT, YOLOP/YOLOPv2, segmentation models.
- audio: Whisper binaries.

Getting the submodule
- Clone main repo, then initialize this submodule:
  - git submodule update --init --recursive

Weights to download (examples)
- TinyLlama base: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Hugging Face), or GGUF quant if using llama.cpp.
- LaneATT checkpoints (culane/tusimple/llamas) from upstream releases.
- YOLOP / YOLOPv2 weights from their official repos.
- Whisper ggml/onnx binaries (if needed).

Storage
- Keep large files on fast local disk. They are ignored by git; stash them under the appropriate subfolders (llm/, vision/, audio/).

