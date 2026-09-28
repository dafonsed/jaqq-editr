# Speech editing audit and fixes

Date: 2026-09-28. This report describes the source in this repository.
The supplied Downloads folder is unchanged. The pasted audit brief was used as the acceptance checklist; historical claims in project documents were checked against the code.

**Result: confirmed code defects fixed and offline regressions pass; the whole editor is NOT VERIFIED.** No problematic source clips were supplied. No real speech was listened to or real edited footage watched. The synthetic render below verifies frame/sample mapping, not speech quality or Resolve's renderer.

## Confirmed causes and changes

| Cause | Consequence | Implemented change |
|---|---|---|
| `dialogue_cut` and `attempt_review` used approximate wording/content overlap before semantic checks | Distinct facts could disappear before meaning review could protect them | Exact replacement/prefix evidence and explicit repair cues now gate deterministic deletions; paraphrases wait for contextual review |
| Number/negation guards were incomplete and inconsistent between passes | Spelled quantities and contractions could evade safeguards | Shared `speech_safety` covers quantities, ordinal words, Unicode apostrophes, negation, quoted examples, confidence and supplied speaker labels |
| Cut-off chains were handled as isolated words | An abandoned opening could survive, or a repaired sentence could break | Full-span restart/correction handling across adjacent transcript segments; ambiguous unique context is kept and flagged |
| Saved text preferences could override current evidence | A choice from a previous recording could authorize a different edit | Current information coverage remains required; saved preferences do not independently authorize deletion |
| The ASR text was treated as nearly complete evidence | Omitted or badly aligned speech could disappear from selection | Raw and recovered transcripts remain inspectable; waveform/transcript disagreements and suspect alignment preserve source spans for review |
| Silence removal used a fixed RMS ceiling of .0008 and a .8-second minimum | Audible room tone prevented dead-space removal; different pause contexts got the same treatment | Transcript-aware room-tone evidence plus separate pause budgets and explicit exemptions |
| Initial phrase grouping removed gaps before later pacing could consider them | Pacing settings could not affect already-separated sentences | Reliable word-free gaps are connected for pause evaluation, without crossing rejected speech or explicit protected spans |
| Edge protection did not subtract exclusions inside a selected interval | A later pass could restore failed-take material | All rejection barriers are enforced before word expansion and frame quantization |
| Waveform edges could trim quiet endings and ignore the last partial analysis window | Word endings could be clipped | Aligned surviving words constrain acoustic refinement; partial windows are retained |
| Python rounded frames differently from generated Lua | Half-frame boundaries could move on export | One integer frame map is serialized and consumed by Resolve; every clip's position, duration and source offset are asserted |
| Audio sidecar caches lacked completed-file integrity checks | Stale/truncated intermediates could be reused | Versioned source identity, atomic files, AAC payload/timestamp checks, and cache hash validation |
| WAV caches trusted only their headers | A truncated file with a valid header was reused | Actual sample bytes are validated; incomplete WAVs are rebuilt |
| Extraction ignored common source timestamp origins and omitted resampler flushing | Analysis timing/tails could disagree with source picture | Video-origin-relative extraction, resampler flushing, missing-timestamp failure, exact sample counts |
| First-run directories were assumed to exist | A fresh source checkout could fail before analysis | Parent creation in extraction, JSON writes, Resolve preflight and UI error logging |
| Import success was called `verified` | The UI/report implied rendered quality had been checked | Separate timeline/render/listening states; UI explicitly calls the result a draft |
| The launcher pointed to a dated prebuilt executable | Editing source did not update the launched application | Launcher now runs the audited source using configured/local Python |

## Pipeline traced

| Stage | Active implementation and behavior |
|---|---|
| Launch/UI | `START CUT REVIEW.cmd` → `bootstrap_runtime.py` → `launch_cut_review.py` → `automatic_review.AutomaticWindow`/`Worker`; isolated runtime setup precedes UI imports; selected microphone and Natural/Balanced/Tight pacing reach `automatic_cut.plan` |
| Ingest | `automatic_cut.probe` uses video duration/rate and audio stream metadata; source size/mtime identity is checked across analysis and before export |
| Audio preprocessing | `analyze_speech.extract`: selected track → 16 kHz mono PCM16 analysis WAV, aligned to video start; original media is unchanged |
| Transcription | Existing local faster-whisper model (preferred `speech-small.en`), CPU int8, word timestamps, beam 5, VAD; verbatim prompt and previous-text context requested. This is not a guarantee that ASR preserves stutters |
| Hidden speech recheck | `speech_recheck.recover` splits suspect stretched-word windows for another decode. Its existing recovery is limited; `speech_evidence.audit` separately protects unexplained sound and uncertain alignment |
| Transcript quality | `transcript_quality.clean` removes clearly collapsed ASR text from semantic consideration. Its original source span remains protected by the evidence audit |
| Visual/action selection | `combat_detection`: local YAMNet gunfire classes and sampled CLIP gameplay/menu checks; `action_sequences` connects supported combat beats. It does not understand arbitrary demonstrations |
| Speech repairs | `dialogue_flow`, `dialogue_cut`, `attempt_review`, `speech_safety`: supported stutters/false starts/corrections, complete take selection and conservative preservation |
| Contextual meaning | `script_review` and `contextual_takes`: sentence candidates in a 60-second neighborhood, MiniLM retrieval, directional DeBERTa NLI, local Qwen proposal/information-loss audit. Surrounding before/between/after text is supplied to Qwen. This is nearby retake review, not whole-video story analysis |
| Pacing | `pause_cleanup`: protected word spans, acoustic pause evidence, context budgets, safe gap bridging, explicit exemptions and final long-gap flags |
| Boundaries | `speech_edges.refine_edges`, then `cut_integrity.protect_words` before and after script review; rejected spans remain barriers |
| Timeline | `review_core.build_frame_map`/`export_review`: contiguous output frame positions; main picture and separate linked source audio; per-clip Resolve assertions |
| Smoothing | No automatic fades/crossfades/room-tone synthesis in the active export. Hard joins remain a listening-review limitation |
| Captions/effects | The current automatic path deliberately excludes caption, motion and SFX modules; creates a new source-only timeline. It does not ripple-edit an existing multitrack timeline |
| Export | `automatic_cut.send` prepares sidecars, writes plan/review/manifest, creates a new Resolve timeline and DRT backup. It does not render a finished video file |

Every automatic run has a new UUID analysis namespace; transcription, waveform and action analysis are fresh. Audio sidecars can be reused only after source-identity and content checks. The UI does not resume partial jobs. Errors are visible and do not return a successful semantic plan. Semantic coverage reports distinguish completed, incomplete and not-applicable stages; no eligible candidate is not described as a whole-video semantic review.

## Pacing controls

Values are seconds of intended retained pause/handle material. Word safety, protected context and frame rounding can retain more; final long gaps are audited rather than silently declared clean.

| Context | Natural | Balanced (default) | Tight |
|---|---:|---:|---:|
| Before useful speech | .120 | .075 | .060 |
| After useful speech | .180 | .120 | .090 |
| Hesitation inside sentence | .400 | .280 | .200 |
| Sentence break | .650 | .450 | .300 |
| Annotated topic break | .950 | .700 | .500 |
| Restart gap | .240 | .200 | .160 |
| Minimum removal | .080 | .060 | .040 |
| Surviving word-free span audit threshold | 1.200 | .900 | .700 |

`pause_settings()` returns the complete configuration, saved in the plan. Topic breaks require `pause_after='topic'` on a preceding word; the code does not invent topic labels. The API accepts `protected_pauses=[{'start': ..., 'end': ..., 'reason': ...}]` for dramatic/comedic pauses or visual demonstrations. Those annotations are protected during selection as well as trimming. There is not yet an annotation UI.

Room tone is eligible only with speech/noise separation and reliable nearby word context. Missing/uncertain context permits conservative near-silence cuts only. Aligned quiet words, selected gameplay, explicit intentional spans, and uncertain source audio take priority over tight pacing. Numerical heuristic confidence values are not calibrated probabilities of editorial correctness.

## Before/after evidence

These are **synthetic word timestamps and waveform fixtures**, not recordings of the user. Word-fixture output coordinates below describe the speech-selection stage before final frame protection/rendering. Full per-word annotations are in `analysis/verification/speech-repairs.json`.

| Fixture | Rejected source | Retained content and source → output time |
|---|---|---|
| `I wa— I wa— I wanted to show you this.` | .925–1.760 | Successful opening begins at source 1.800 → output .040; full successful phrase retained |
| `I I think this works.` | .925–1.160 | Second `I` at 1.200 → .040; rest of sentence retained |
| `What you need to— what I recommend is changing this setting.` | .925–1.760 | `what I recommend...` at 1.800 → .040 |
| `It costs fifteen— sorry, fifty dollars.` | 1.360–1.760 | `It costs` retained; corrected `fifty` at 1.800 → .475; no fifteen in selected text |
| `You should enable it— actually, don’t enable it.` | See annotation | Complete corrected negative instruction retained; its `don't` survives |
| `This is really, really important.` | None | Both emphasis words retained |
| Quoted/demonstrated `I I...` | None | Demonstration retained |
| Correction with unique `First save the file...` context | None | Whole ambiguous repair preserved and flagged, avoiding loss of the unique instruction |
| Failed opening and successful continuation in adjacent transcript chunks | .925–1.360 | Successful `I` at 1.400 → .040 |

Room-tone reproduction: source words at [1,2] and [4,5] seconds in a six-second waveform with .003-amplitude room tone. Original pause cleanup left [0,6] intact. Balanced cleanup retains [0.925,2.276923] and [3.826923,5.120], totaling **2.645 seconds**, with **.450 seconds** between the two spoken words. The final frame-aligned plan may add up to a frame per edge.

Actual synthetic media test: a 90-frame MP4 containing changing frame colors and two separate AAC sine-wave tracks was mapped to a lossless 60-frame, two-second output. The test deliberately includes reordered and repeated ranges to stress coordinate mapping; automatic speech planning still preserves chronology. All decoded output frames and both 96,000-sample audio tracks matched the intended source slices. Packet payloads and presentation times were verified through sidecar remux. A source beginning at timestamp 10 seconds also passed relative extraction/crop checks. This test renderer uses the shared map but is **not Resolve**.

Artifacts: `analysis/verification/synthetic-export/synthetic-source.mp4`, `synthetic-mapped-output.mkv`, `synthetic-verification.json`, and `plan/review.json`.

## Acceptance status

PASS below is deliberately limited to the named code/fixture evidence.

| Area | Status | Evidence / remaining requirement |
|---|---|---|
| Annotated clear stutters, abandoned openings, full retakes and adjacent-chunk repairs | PASS | Deterministic timestamp fixtures; actual ASR/listening still unverified |
| Corrected quantities/negations and unique context | PASS | Positive repair and negative preservation regressions |
| Intentional repetition and quoted mistakes | PASS | Preservation fixtures; real speaker intention remains contextual |
| Room-tone pauses and configurable pacing | PASS | Acoustic fixtures cover intensity, sentence/topic/restart and protected spans |
| Quiet-word boundary protection | PASS | Low-amplitude ending and frame-barrier regressions; no real consonant listening |
| No unintended output timeline gaps | PASS | Integer-map invariants and decoded synthetic frame/sample continuity; actual Resolve result unverified |
| Cache/retry/settings integrity | PASS | Fresh analysis twice, truncated WAV/corrupt sidecar rebuild, settings routing, changed source/plan rejection |
| Real transcription and paraphrase-model accuracy | NOT VERIFIED | Model assets are absent from this source checkout; semantic policy tests use explicit mocked scores |
| Real-source stutters and repaired joins sound natural | NOT VERIFIED | Source recordings unavailable; no listening or waveform-only approval claimed |
| Crossfade/click/pop/lip-sync quality | NOT VERIFIED | No production smoothing implemented; real rendered joins must be assessed |
| Actual Resolve import and final rendered export | NOT VERIFIED | Resolve scripting executable exists here, but no user-media import/render was executed |
| Caption/B-roll/graphics/music/chapter ripple synchronization | FAIL | Active architecture is source-only and does not carry these tracks or captions through edits |
| Whole-video opening/payoff/tangent/ending structural review | FAIL | Nearby retake review cannot establish whole-story editing; no global content/visual understanding implemented |
| Automatic multi-speaker handling | NOT VERIFIED | Supplied speaker labels are guarded, but the active transcription path does not perform diarization |
| Every retained long pause has an established editorial reason | NOT VERIFIED | Unexplained spans are visibly flagged; uncertain sound is not assigned a fabricated reason |
| Entire editor meets all requested acceptance criteria | NOT VERIFIED | Real recordings, model runs, listening, visual review and Resolve rendering remain necessary |

These are editorial improvements. There is no audience-retention measurement or claim of increased viewership.

## Running and reviewing the source

`START CUT REVIEW.cmd` now runs this source, so source changes cannot be hidden by the old dated executable. It uses `RETENTION_PYTHON`, a local `.venv`, the available Codex Python runtime, or Python on PATH to bootstrap a project-local `.venv`. First-run installation uses `requirements-runtime.txt` and checks native imports before starting the GUI. It does not rely on an ignored `.runtime-deps` directory being present in a GitHub download. Existing separately packaged applications have not been rebuilt or installed.

The source download omitted `.model-cache`, `.runtime-deps` and private `analysis`/media assets. Initial testing used local `.runtime-deps`; the repaired launcher installs the complete Python runtime, including `faster_whisper`, into `.venv`. The speech/embedding/NLI/Qwen/YAMNet/CLIP model assets were not downloaded. Restore the matching model assets from the existing application before attempting real processing. No new model or inference architecture was added.

Run the explicit offline suite with a Python interpreter that can access this copy's dependencies:

```powershell
.\.venv\Scripts\python.exe run_offline_checks.py
```

Do not indiscriminately execute all historical `test_*.py` or deployment scripts: several operate on live Resolve projects or private example paths. `run_offline_checks.py` lists 19 reviewed offline scripts and writes `analysis/verification/offline-checks.json`; it does not invoke Resolve or download models. Startup coverage includes failed native imports, pip/setup failures, isolated dependency precedence, paths containing spaces, and preservation of GUI failure exit codes. Actual first-run installation and GUI startup were also checked from a clean source-only copy with no `.runtime-deps` or pre-existing `.venv`.

Each new real export writes `automatic-plan.json`, `edit-manifest.json`, `review.json`, `script-review.json`, `selected-script.txt`, Resolve assertion output and a DRT backup. They expose removed/retained source spans, reasons/evidence, uncertainty, actual output coordinates and separate verification states. Original media and existing timelines are preserved.

When the problematic recordings are available, the remaining acceptance work is to annotate the actual defects before rerunning, compare original/rendered speech around every repaired join, assess ambiguous takes and visual demonstrations, and inspect the real Resolve export for synchronization and artifacts. Model-scored paraphrases and hard joins must not be approved on these offline tests alone.
