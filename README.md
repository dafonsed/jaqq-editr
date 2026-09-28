# JAQQ Editor

Speech and pacing editor with local analysis, optional OpenAI edit proposals, and draft timeline export for DaVinci Resolve.

## Start on Windows

1. Keep the downloaded source files together in a folder.
2. Install **64-bit Python 3.12 or newer** if Python is unavailable. The launcher can also use a local Codex Python runtime.
3. Double-click **START CUT REVIEW.cmd**.

The first launch creates `.venv` in this folder, installs the packages listed in `requirements-runtime.txt`, and checks their actual imports before opening the editor. Internet access is required for this first setup. Subsequent launches reuse the verified runtime. Packages are not installed into your system Python.

Do not launch individual `.py` files from Explorer. The START command prepares the environment they need.

The editor defaults to local analysis when no OpenAI API key is configured. On a fresh download, click **Set up local AI** to download approximately **3.8 GB** of speech, semantic, editorial and gameplay models. Progress appears in the window; interrupted downloads can be retried. Files are pinned and checksum-verified before use. Local analysis keeps recordings on your computer. Models are cached in `.model-cache` and are not stored in Git. Once setup completes, choose a recording, select its commentary track, and create the cut.

OpenAI mode requires `OPENAI_API_KEY` and sends the selected transcription audio and transcript/context for model proposals. It opens a review window to compare, restore, and adjust proposed cuts before exporting. Configuration, data flow, evaluation results, and remaining limitations are documented in [AI_DECISION_SYSTEM.md](AI_DECISION_SYSTEM.md). Live OpenAI quality has not been validated in this checkout because no API key was available.

If Resolve's external scripting API is available, open a project before exporting for direct timeline creation. Otherwise the editor saves a draft folder: click **Open draft files**, then in Resolve choose **File → Import → Timeline** and select **Automatic cut.xml**. Source media must remain accessible. The XML fallback was imported into Resolve and exported back for frame-range verification. Export creates an editable draft; it does not render a finished video.

## Startup troubleshooting

Run these commands from the source folder:

```powershell
# Install/check packages without opening the editor
& '.\START CUT REVIEW.cmd' --setup-only

# Reinstall a damaged local runtime
& '.\START CUT REVIEW.cmd' --repair --setup-only

# Open the real GUI, verify startup, and exit
& '.\START CUT REVIEW.cmd' --launch-check

# Run the offline regression suite using the same runtime
.\.venv\Scripts\python.exe run_offline_checks.py
```

Setup diagnostics are in `analysis/runtime-setup.log`; normal GUI output and native crash diagnostics are in `analysis/application.log`, and Python startup errors are in `analysis/launch-error.log`. The normal launcher waits for the visible window to acknowledge startup; opening it again restores the existing window. To select another Python installation, set `RETENTION_PYTHON` to its executable path before running the START command.

The source launcher uses its isolated `.venv` rather than an old `.runtime-deps` folder, so stale native packages cannot override the checked installation.
