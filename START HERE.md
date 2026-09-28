# Retention Cut

See [AUDIT_AND_FIX_REPORT.md](AUDIT_AND_FIX_REPORT.md) for the current speech/pacing fixes, runtime requirements, verification results and remaining limitations. The launcher now runs source; an old prebuilt executable does not include these changes.

Open a project in Resolve, open Cut Review, choose your recording and microphone track, then click Create automatic cut.

Retention cutting only: original picture and linked voice/game audio. No captions, sound effects, memes, zooms, motion graphics or effect markers.

Each run creates a new RETENTION timeline, a backup and cut decisions in exports. Existing timelines and recordings remain intact. There is no runtime target or removal quota.

Before export, a local rule-based pass reviews the assembled dialogue in playback order. It removes clear adjacent repeated takes when no gameplay event is affected and flags uncertain joins. Each export includes selected-script.txt with playback timestamps and script-review.json with changes and flags. This is a structural check, not an AI semantic verdict that every sentence makes sense.

Review dialogue joins and action context. Transcription may miss or mistime words; text matching does not fully understand the story. Keep the app, .runtime-deps, .model-cache and analysis folders together.
