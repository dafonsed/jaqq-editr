# JAQQ Editor

Local speech and pacing editor that creates a new draft timeline in DaVinci Resolve.

## Start on Windows

1. Keep the downloaded source files together in a folder.
2. Install **64-bit Python 3.12 or newer** if Python is unavailable. The launcher can also use a local Codex Python runtime.
3. Double-click **START CUT REVIEW.cmd**.

The first launch creates `.venv` in this folder, installs the packages listed in `requirements-runtime.txt`, and checks their actual imports before opening the editor. Internet access is required for this first setup. Subsequent launches reuse the verified runtime. Packages are not installed into your system Python.

Do not launch individual `.py` files from Explorer. The START command prepares the environment they need.

The runtime installer installs **Python packages only**. Speech, semantic and gameplay model weights are separate `.model-cache` assets and are not included in this repository. Restore the model folder from the existing app before processing recordings. See [the audit report](AUDIT_AND_FIX_REPORT.md) for the required models and verified limitations. Installing the Python runtime does not verify a real edit or a Resolve render.

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

Setup diagnostics are in `analysis/runtime-setup.log`; GUI errors are in `analysis/launch-error.log`. To select another Python installation, set `RETENTION_PYTHON` to its executable path before running the START command.

The source launcher uses its isolated `.venv` rather than an old `.runtime-deps` folder, so stale native packages cannot override the checked installation.
