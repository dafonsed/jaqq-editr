# Retention Cut

See [AUDIT_AND_FIX_REPORT.md](AUDIT_AND_FIX_REPORT.md) for the current speech/pacing fixes, runtime requirements, verification results and remaining limitations. The launcher now runs source; an old prebuilt executable does not include these changes.

Double-click **START CUT REVIEW.cmd**. On first launch it creates and verifies a local `.venv`, including PySide6, before opening the window. Setup needs internet access and 64-bit Python 3.12 or newer; [README.md](README.md) has repair commands. If the window shows **One-time setup needed**, click **Set up local AI** to download and verify the approximately 3.8 GB model files. Editing is enabled when those files are ready.

Choose your recording and microphone track, then click Create automatic cut. Local mode works with the installed models. Optional OpenAI mode requires `OPENAI_API_KEY` and lets you review proposals before export; see [AI_DECISION_SYSTEM.md](AI_DECISION_SYSTEM.md).

Retention cutting only: original picture and linked voice/game audio. No captions, sound effects, memes, zooms, motion graphics or effect markers.

For direct export, open a project in Resolve with its external scripting API available. Otherwise the editor saves a draft folder: click Open draft files, then use Resolve's File → Import → Timeline and select Automatic cut.xml. Successful direct import creates a new RETENTION timeline and backup. Existing timelines and recordings remain intact. There is no runtime target or removal quota.

Before export, a local rule-based pass reviews the assembled dialogue in playback order. It removes clear adjacent repeated takes when no gameplay event is affected and flags uncertain joins. Each export includes selected-script.txt with playback timestamps and script-review.json with changes and flags. This is a structural check, not an AI semantic verdict that every sentence makes sense.

Review dialogue joins and action context. Transcription may miss or mistime words; text matching does not fully understand the story. Keep the app, .venv, .model-cache and analysis folders together.
