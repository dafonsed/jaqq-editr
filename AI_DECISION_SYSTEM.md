# AI retention decision system — implementation and verification

Implemented in the existing editor on 2026-09-28. The original cutting/export
engine remains responsible for the timeline. OpenAI output is a proposal; it
never directly supplies executable cut timestamps.

**Verification status:** the offline suite passes, and actual synthetic speech
was processed and rendered. No OpenAI API key was available, and the user deferred
providing a real recording. Live gpt-transcribe/Sol/Astra accuracy, latency, cost,
listening quality and human preference remain **NOT VERIFIED**. Test responses
and fixture replay are explicitly labeled; they are not evidence of model quality.

## 1. Audit of the previous pipeline

`automatic_review.Worker` called `automatic_cut.plan` and immediately sent its
result to Resolve. `analyze_speech.extract` decoded the selected microphone track
to source-aligned 16 kHz mono PCM. `faster-whisper` supplied segment and word times
and recognition probabilities; `speech_recheck` revisited unusually stretched
word spans. There were no reliable automatic speaker labels.

`dialogue_cut` / `dialogue_flow` applied lexical speech repairs during initial
selection. `speech_edges` adjusted handles using 10 ms waveform energy, and
`pause_cleanup` classified and shortened word-free gaps. `contextual_takes` and
`script_review` compared nearby sentences using local MiniLM, NLI and Qwen.
`combat_detection` could preserve gameplay from a separate audio track. No cloud
OpenAI API, secret handling, retries, API billing ledger or resumable API stages
existed. Every local analysis used a fresh namespace.

This order meant contextual review happened after some deletions. Short
corrections and failed starts could miss the sentence-comparison gate; lexical
similarity was inadequate for paraphrased retakes. Silence evidence could not
establish a joke, reveal or visual demonstration. The current full local baseline
confirmed the short repair gap: multiple examples were ineligible for semantic
review because they had fewer than two eligible sentence spans.

`review_core.build_frame_map` already supplied shared integer, half-open source
and output ranges. `review_media` checked video/audio origins and extracted linked
audio. `automatic_cut.send` created a Resolve draft and timeline verification,
not a final rendered-video or listening verification. The automatic launcher had
no proposed-cut review; the older `cut_review` UI had separate manual controls.
Captions/motion/SFX helpers existed, but the retention-only export deliberately
did not add those effects to the new draft.

## 2. Files and behavior changed

| File | Responsibility |
|---|---|
| `openai_editor.py` | Fixed-origin HTTPS transport; Responses Structured Outputs; multipart transcription; timeout/retry/cancellation; validated caches and durable usage ledger |
| `ai_transcription.py` | Overlapping gpt-transcribe chunks; explicit correspondence to existing acoustic word timing; untimed cloud text and uncertainty |
| `ai_decisions.py` | Candidate retrieval without deletion, contextual windows, Sol/Astra decisions, schema and editorial/geometry validation |
| `ai_pipeline.py` | Source extraction, local timing, protection of uncertain speech and other audio tracks, checkpointed plan, restoration and export revalidation |
| `automatic_review.py` | OpenAI/local selection, key-aware defaults and review before OpenAI export |
| `ai_review_ui.py` | Reasons/confidence/model timeline; before/after playback; restore/reapply/undo; validated boundaries; explicit Resolve export |
| `review_render.py` | Real review encoding from the existing frame map, separate audio tracks, mapped SRT, fresh render transcription and measurable QA |
| `automatic_cut.py` | Revalidate AI decisions before the existing export and write decision/caption artifacts |
| `pause_cleanup.py` | Accept `aggressive` as the historical `tight` preset alias |
| `evaluation_cases.json`, `generate_ai_eval.py`, `evaluate_ai_pipeline.py`, `evaluate_local_baseline.py`, `verify_ai_review_media.py` | Reproducible synthetic fixtures, complete local baseline, fixture/live decision evaluation and rendered-media inspection |
| `test_openai_editor.py`, `test_ai_transcription.py`, `test_ai_decisions.py`, `test_ai_pipeline.py`, `test_ai_review_ui.py`, `test_ai_review_render.py`, `run_offline_checks.py` | Transport, safety, integration, UI and actual encode/decode regressions |

The coordinated startup/Resolve work is documented separately in
`AUDIT_AND_FIX_REPORT.md` and `MODEL_SETUP.md`. It adds manual XML/DRT draft
handoff when external Resolve scripting is unavailable. Such a fallback is
explicitly unverified until imported; it is never reported as a verified render.

OpenAI mode retains the complete source until approved deletions pass validation.
The review screen can restore an accepted cut, shrink its boundaries, undo a
change and render the resulting context or full edit. It cannot turn a rejected
speech deletion into an arbitrary timestamp cut. Persisted review files include
the source identity and cut signature. Render evidence survives unchanged export
and is invalidated by a change in geometry.

## 3. Exact models, escalation and pacing

* `gpt-transcribe`: `/v1/audio/transcriptions`, verbatim speech instructions.
* `gpt-6-sol`: `/v1/responses`, medium reasoning, strict JSON Schema.
* `gpt-6-astra`: `/v1/responses`, high reasoning, only eligible escalations.

Actions are `KEEP`, `REMOVE`, `SHORTEN_PAUSE`,
`REPLACE_EARLIER_TAKE_WITH_LATER_TAKE`, and `NEEDS_REVIEW`. Each decision includes
source bounds, authorized word IDs, intended surviving word IDs, required context,
reason, confidence and uncertainty fields. Requested/effective provider models,
request IDs, prompt versions and cache provenance are recorded. A cached earlier
Astra response is identified as cached; failed calls and fixture adapters cannot
claim a completed Astra review.

Non-pause candidates escalate for confidence below 0.90, explicit ambiguity or
`NEEDS_REVIEW`, possible meaning/quantity/negation changes, or missing decisive
audio/visual context. Routine pause candidates do not escalate. Unambiguous KEEP
does not escalate. Astra cannot override hard protection of quoted/demonstrated
speech, conflicting speaker labels, uncertain timing, or unresolved meaning.

Each candidate belongs to one 30-second owner window with 12-second context on
each side. Mandatory replacement/context words extend that window when needed.
Batches contain at most four candidates and normally at most 600 context words.
Cloud transcript text is supplied separately as untimed evidence. Overlapping
decisions that would erase a surviving take are preserved for review. Identical
removals have one executable authority, so restoring one cannot leave a duplicate
cut active.

| Profile | Hesitation / sentence / topic pause budget | Leading / trailing handle | Speech confidence threshold |
|---|---|---|---|
| Natural | .40 / .65 / .95 s | .12 / .18 s | .96; preserves conversational fillers |
| Balanced | .28 / .45 / .70 s | .075 / .12 s | .92 |
| Aggressive (`tight` in older saved settings) | .20 / .30 / .50 s | .06 / .09 s | .88 |

All modes retain the same meaning/timing safeguards. Validated pause proposals
require at least .80 confidence. These are contextual budgets, not a command to
delete every silence. Explicit protected pauses and selected visual/action
intervals veto removals. Local voice activity on other audio tracks protects
possible voice-chat/participant speech; it is not speaker identification.

## 4. Timing, boundaries and operations

The transcription request does not request unsupported exact word timestamps.
The existing local speech recognizer measures word timing against the waveform.
Cloud text is matched monotonically to those words; no missing timestamp is
interpolated. Ambiguous repetition mappings and text mismatches remain explicit.
Cloud-only words protect the owning source chunk because their location is
unknown. This can retain a large region conservatively.

Transcription audio chunks are up to 480 seconds with eight seconds of overlap;
disjoint ownership prevents duplicate words. Local acoustic timing and successful
API responses persist under `analysis/openai/<source-identity>-mic<track>/`.
Cache identity includes request content, model, schema, reasoning and prompt
version. Changed source identity invalidates reuse. Failed/incomplete requests
do not become a successful empty plan; successful previous chunks can resume.

The validator checks finite values, media duration, known candidate IDs, exact
authorized source words and supplied bounds, contiguous selection, surviving
word IDs, required context, quote/speaker/fact safeguards, protected intervals,
replacement conflicts and minimum retained clip spacing. Execution edges are
derived from aligned words and actual waveform minima, then quantized to source
frames. High-energy joins without safe handles are review-only. Validation also
checks the final padded interval, not just the selected word centers. A partial
container-duration tail is handled with the same final complete video frame used
by the renderer, avoiding artificial one-frame leftovers.

Audio, video, caption times and generic annotation/effect spans share that
canonical frame map. Export writes `ai-decisions.json`, `output-words.json` and
`aligned-captions.srt`. Captions are sidecars, not automatically burned or imported
into a subtitle/effect track. Original media/effects baked into the picture remain
synchronized; pre-existing unrelated Resolve timeline edits are not transplanted.

The review renderer performs actual encoding, then decodes the result to check
frame count/timestamps, all audio-track PTS and duration, silence, sample clipping,
join amplitude changes and large visual discontinuities. A separate local ASR
pass compares rendered words with expected retained words; it is a diagnostic,
not a distinct ground-truth model. Visual jump flags suggest inspecting a crop,
transition or B-roll in Resolve rather than silently applying an effect.

Default audio joins remain the original hard joins, matching the existing Resolve
export. The review renderer exposes optional 0–10 ms waveform-gated handle fades
without changing duration, but the UI does not turn these on or claim they are
production crossfades. Automatic production overlap/crossfades are not implemented;
joins that still need smoothing require listening and a Resolve adjustment.

Only the Python process reads `OPENAI_API_KEY`. The key is never placed in UI
preferences, plan JSON, client code or request-error logs. API calls use the fixed
OpenAI HTTPS origin and refuse redirects. Default timeout is 60 seconds, with up
to three retries for rate limits/transient failures, bounded exponential/Retry-After
backoff and cancellation checks. Incomplete/refused/invalid replies preserve the
affected source. Failed transcription preserves all source. Errors are sanitized.

## 5. Actual evaluation results

Nine local Windows synthetic-speech clips total **58.533 seconds**. Actual local
Whisper timing, extraction, candidate validation and video/audio rendering ran.
The complete existing local pipeline also ran, including actual MiniLM/NLI/Qwen
when eligible. The new pipeline used explicit expected-output fixture responses
because an API key was absent. **This table tests execution of proposals; it is
not a live comparison of local models against Sol/Astra.**

| Clip | Source | Complete local output | New fixture-proposal output | Observed result |
|---|---:|---:|---:|---|
| Repeated start | 5.133 s | 3.767 s | 2.933 s | Local retained `I was. I was…`; new plan retained the later wording |
| Sentence restart | 5.467 s | 4.467 s | 4.567 s | Uncertain word timing preserved the unfinished start for review |
| Blue/green correction | 5.000 s | 4.033 s | 4.100 s | Uncertain timing preserved the wrong-slot correction for review |
| Full sentence retake | 9.633 s | 6.933 s | 4.067 s | Local retained both; fixture proposal kept the later take |
| Filler | 4.867 s | 3.867 s | 3.933 s | Low-confidence `um` preserved; fresh ASR omitted it and flagged disagreement |
| Long dead air | 8.433 s | 3.300 s | 3.400 s | Dead air shortened; no remaining >.9 s quiet-span flag in new render |
| Intentional reveal | 6.200 s | 4.500 s | 6.200 s | Annotated reveal timing preserved |
| Ambiguous correction | 7.200 s | 5.433 s | 5.567 s | Meaningful speech preserved |
| Quoted repetition | 6.600 s | 5.600 s | 5.700 s | Quoted repetition preserved |

Final new rendered checks: **9/9 frame/audio timing and source-plan checks PASS**;
**8/9 fresh ASR word comparisons match**. The filler difference is not proof that
audio was lost. No clipping, AV timestamp drift/gaps or large sample-discontinuity
join flags were detected. Sampled decoded source/render pictures match (maximum
mean RGB difference about .302/255); the contact sheet was visually inspected.
These are measured checks, not a listening or perceptual lip-sync certification.

Residual quiet spans remain flagged: correction 1.16 s, restart 1.20 s and the
intentional-pause fixture 3.14 s plus 1.02 s. Stutter, retake and dead-air renders
have no remaining >.9 s quiet span. Intentional quiet is not automatically an error.

Evidence kept locally (generated media are ignored by Git):

* `analysis/ai-evaluation/evaluation-results.json` — final fixture integration metrics.
* `analysis/ai-evaluation/run-1790635875759669200/` — immutable final new renders.
* `analysis/ai-evaluation/run-1790635875759669200/independent-review/summary.json` — fresh transcription, source/plan checks and picture correspondence.
* `analysis/ai-evaluation/run-1790635875759669200/independent-review/source-render-contact-sheet.png` — decoded source/after inspection.
* `analysis/ai-evaluation/local-baseline-1790635750553247100/report.json` and `SUMMARY.md` — full local baseline, measured model calls and render QA.
* `analysis/verification/offline-checks.json` — all **30 safe check scripts pass**.

Policy regressions additionally cover window boundaries, invalid schema/timestamps,
quoted and multiple-speaker words, changed meaning/negation, duplicate cut restore,
API refusal/retry/resumption, other-track speech protection and caption/effect
mapping. No blanket collection of historical live-Resolve tests was used.

## 6. Cost and processing measurements

There were **zero live OpenAI API calls and $0 API expenditure** in this
verification. Actual Sol/Astra token usage, escalation rate, transcription price
and cloud latency on these clips are unmeasured, not zero-cost forecasts.

The full local baseline analyzed 58.533 s of speech in **81.562 s** (about
83.61 seconds per source minute); its complete render/re-transcription evaluation
took 98.047 s. One actual Qwen call took 55.016 s (469 prompt / 507 completion
tokens). NLI coverage of the retake was .9126, below its .92 acceptance gate.

The new path's measured cold acoustic-timing work across these clips was
**16.831 s** (about 17.25 seconds per source minute). This excludes real cloud
inference. The final cached fixture replay spent .047 s in proposal orchestration
and 1.688 s rendering its nine outputs; independent rendered inspection took
11.39 s excluding shared model initialization. Warm replay timings must not be
presented as live end-to-end processing speed.

Production plans track per-run and cumulative per-video transcription/Sol/Astra
requests, tokens, cache hits, HTTP attempts, elapsed time, cuts and removed duration.
Costs use a dated configurable code rate table and are labeled estimates. Unknown
usage, unreadable accounting records or possibly billed failed/retried requests
retain an unknown total instead of reporting a false zero.

Official list-rate snapshot checked 2026-09-28: transcription $0.0045/audio minute;
Sol $2 input / $0.20 cached input / $10 output per million tokens; Astra
$10 / $1 / $50 respectively. The source pages are linked below.

## 7. Remaining failure cases and verification limits

* Natural low-energy speech may remain unedited: synthetic `Click` scored .59,
  filler `um` .17 and restart words .56/.29. The system preserves these phrases
  instead of assuming their word edges are exact.
* ASR may omit a stutter or invent/omit a filler. Cloud/local disagreement can
  preserve an entire owner chunk. Reliable forced alignment beyond the existing
  acoustic timing is not yet provided.
* Same-track speakers are not automatically diarized. Supplied labels and
  other-track VAD are protected, but a repeated `yes` from two speakers on one
  track still needs human review.
* Jokes, visual demonstrations, subtle emphasis, room noise and breath quality
  cannot be established by text/waveform statistics alone. Protect those spans
  or restore proposed cuts before export.
* Conflicting multi-retake chains remain review-only. A preserved candidate can
  leave a repeated introduction or a longer pause; flags are intentional.
* Hard joins can still sound unnatural even when numerical checks pass. Listening,
  real microphone consonants, production crossfades, perceptual lip-sync, imported
  captions/effects and human preference remain unverified.
* The linked-audio Resolve path retains the existing media constraints, including
  conservative rejection of unsupported variable frame rate/audio layouts/origins.
* Source aliases and future model changes can change a fresh response. Stored
  prompts, schemas, requested/effective models, source identity and responses
  reproduce the saved proposal, not a guarantee of identical future inference.

## 8. Configuration and repeating verification

Use the existing START launcher. With no API key, the launcher defaults to Local;
the OpenAI option remains visible. OpenAI mode requires the local speech timing
model plus an API key with access to the three exact models above. It does not
require the local Qwen semantic model. Select the commentary track explicitly when
multiple audio tracks are present.

Set the key in the launching process, not in chat or source files. For a temporary
PowerShell session (secret input does not enter command history):

```powershell
$editorSecret = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $editorSecret).Password
& '.\START CUT REVIEW.cmd'
```

Restart the editor after changing the environment. OpenAI mode sends selected
commentary audio and transcript context to OpenAI; the UI states this. Private
caches and transcripts remain in the ignored `analysis/` folder. Existing `.env`
files are not automatically read.

From the project directory:

```powershell
.\.venv\Scripts\python.exe run_offline_checks.py
.\.venv\Scripts\python.exe generate_ai_eval.py
.\.venv\Scripts\python.exe evaluate_ai_pipeline.py
.\.venv\Scripts\python.exe evaluate_local_baseline.py
# Real API evaluation; uses the configured key and incurs actual API charges:
.\.venv\Scripts\python.exe evaluate_ai_pipeline.py --live
```

Then evaluate the user's representative recording, listen to before/after joins,
inspect final Resolve output and collect human preferences. Those specific steps
are blocked until the credential and deferred media sample are supplied.

Official API contracts/rates used:
[speech transcription](https://developers.openai.com/api/docs/guides/speech-to-text),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[GPT-Transcribe](https://developers.openai.com/api/docs/models/gpt-transcribe),
[GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol),
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).
