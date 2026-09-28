# Local model setup

The editor's setup worker downloads the models already used by the editing
pipeline. Setup downloads about **3.77 GB** once, keeps the files in the ignored
`.model-cache` directory, and reuses them on later launches. No footage or
transcripts are sent to the model hosts.

`model-assets.json` pins every download to an immutable model commit or release,
the expected byte length, and its published SHA-256 or Git blob SHA-1. The setup
worker verifies every file before installing it. The official llama.cpp archive
is SHA-256 checked before its engine and DLLs are extracted. Readiness receipts
also detect missing, truncated, or changed installed files on later launches.

| Component | Source | Pinned revision |
| --- | --- | --- |
| English speech, faster-whisper small.en | [Systran/faster-whisper-small.en](https://huggingface.co/Systran/faster-whisper-small.en) | `d1d751a5f8271d482d14ca55d9e2deeebbae577f` |
| Sentence embeddings, quantized ONNX | [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) | `751bff37182d3f1213fa05d7196b954e230abad9` |
| Directional meaning checks, float32 ONNX | [Xenova/nli-deberta-v3-small](https://huggingface.co/Xenova/nli-deberta-v3-small) | `6bc2a55c7c0f7e2bc68de60bb248e523e2612abb` |
| Editorial review, Q4_K_M GGUF | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | `bc640142c66e1fdd12af0bd68f40445458f3869b` |
| Local inference engine, Windows x64 CPU | [ggml-org/llama.cpp b10809](https://github.com/ggml-org/llama.cpp/releases/tag/b10809) | `b10809` |
| Audio event classifier | [audiomagic/yamnet-onnx](https://huggingface.co/audiomagic/yamnet-onnx) | `f25b741c2f0bdc6d7e6db24b5fddda23347dbafd` |
| Gameplay/menu visual checks, quantized ONNX | [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32) | `d15189d7028b43f1d3e65039190477f6af591c2a` |

Setup can be stopped and retried. Completed assets are retained. Interrupted
downloads resume when the host supports byte ranges; otherwise they restart
without appending duplicate data. Unverified partial files are never reported
as ready. Network reads time out after 15 seconds, allowing cancellation to take
effect even when the connection stalls. A process lock prevents two editor
windows from modifying the model cache simultaneously.

For diagnostics from an already provisioned source checkout:

```powershell
.\.venv\Scripts\python.exe model_assets.py --status
.\.venv\Scripts\python.exe model_assets.py
```

`status()` is a fast disk-only readiness check. `ensure_models(progress, cancel)`
is the cancellable worker API. It raises `SetupCancelled` when stopped and a
readable exception when setup cannot finish. This setup check validates model
files; inference and whole-pipeline validation are separate checks.
