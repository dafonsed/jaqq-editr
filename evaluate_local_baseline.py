"""Run the complete existing local editor on synthetic evaluation recordings.

Uses actual Whisper/MiniLM/NLI/Qwen inference when their normal production
eligibility rules invoke them. It never calls OpenAI or exports to Resolve.
Every execution writes a new directory; failures remain in the report.
"""
import argparse
from contextlib import contextmanager
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app_paths import ROOT
from automatic_cut import plan
from review_core import complement, map_source_span, save_json
from review_render import render_preview, audit_render, transcribe_render


def normalized(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def text_metrics(expected, actual):
    reference, observed = normalized(expected), normalized(actual)
    return dict(expected=expected, retained_source_text=actual,
        exact_expected_token_match=reference == observed,
        differences=[dict(operation=op, expected=reference[a:b], actual=observed[x:y])
                     for op, a, b, x, y in SequenceMatcher(None, reference, observed, autojunk=False).get_opcodes()
                     if op != "equal"],
        limitation="Source/independent ASR may be wrong; token agreement is not a listening verdict.")


def code_identity():
    files = ("automatic_cut.py", "script_review.py", "contextual_takes.py", "local_editor.py",
             "semantic_review.py", "dialogue_cut.py", "speech_safety.py", "speech_edges.py",
             "pause_cleanup.py", "review_render.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


@contextmanager
def measured_models(events, progress):
    """Instrument calls without changing requests, outputs or execution policy."""
    import contextual_takes
    from local_editor import Editor
    originals = []

    def wrap(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        def measured(*args, **kwargs):
            started = time.monotonic()
            record = dict(model=label, operation=name, status="running")
            if label == "Qwen3-4B-Q4_K_M":
                record["configured_request_timeout_seconds"] = args[0].request_timeout
            events.append(record)
            progress(label + " actual inference started")
            try:
                result = original(*args, **kwargs)
                record["status"] = "completed"
                if label == "Qwen3-4B-Q4_K_M":
                    record["usage"] = result.get("usage")
                    record["finish_reasons"] = [r.get("finish_reason") for r in result.get("choices", [])]
                    if any(reason != "stop" for reason in record["finish_reasons"]):
                        record["status"] = "incomplete"
                return result
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - started, 4)
                progress(f"{label} {record['status']}: {record['elapsed_seconds']:.2f}s")
        setattr(owner, name, measured)

    wrap(contextual_takes, "encode", "MiniLM")
    wrap(contextual_takes, "entailment", "DeBERTa-v3-small NLI")
    wrap(Editor, "_request_json", "Qwen3-4B-Q4_K_M")
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def baseline_transcript(result):
    key = result["source_key"] + "-run-" + result["analysis_run_id"]
    path = ROOT / "analysis" / ("recovered-transcript-" + key + ".json")
    from transcript_quality import clean
    transcript, _ = clean(json.loads(path.read_text(encoding="utf-8")))
    return transcript, str(path)


def write_summary(run, report):
    completed = [c for c in report["cases"] if "qa" in c]
    source = sum(c["source_seconds"] for c in report["cases"])
    analysis = sum(c.get("analysis_seconds", 0) for c in report["cases"])
    lines = ["# Full local baseline on synthetic recordings", "",
        "This run invoked the actual existing `automatic_cut.plan` pipeline. Model calls were measured without changing their inputs or decisions. The comparison pipeline used replayed fixture responses, so this does not measure OpenAI model quality.", "",
        f"Completed {len(completed)}/{len(report['cases'])} cases. Total source: {source:.3f}s. Analysis: {analysis:.3f}s ({analysis/(source/60):.2f}s per source minute). Analysis, rendering and independent ASR: {report['total_seconds']:.3f}s. External API calls and charges: 0 / $0.", "",
        "| Case | Full local output | Fixture pipeline output | Local analysis | Local expected wording | Rendered ASR vs selected words |",
        "|---|---:|---:|---:|---|---|"]
    for case in report["cases"]:
        if "qa" not in case:
            lines.append(f"| {case['id']} | FAILED | — | — | {case.get('error_type', 'Unknown')} | — |")
            continue
        other = case.get("fixture_integrated_comparison", {})
        fixture = f"{other['output_seconds']:.3f}s" if "output_seconds" in other else "unavailable"
        match = "matches" if case["source_text_metrics"]["exact_expected_token_match"] else "differs"
        lines.append(f"| {case['id']} | {case['output_seconds']:.3f}s | {fixture} | {case['analysis_seconds']:.3f}s | {match} | {case['qa']['word_check']['status']} |")
    lines.extend(["", "All word comparisons depend on acoustic ASR and require listening confirmation."])
    filler = next((c for c in completed if c["id"] == "filler"), None)
    if filler and filler["qa"]["word_check"]["status"] == "REVIEW":
        lines.append("The local filler render disagrees with the selected source-word plan under independent ASR; this is flagged, not credited as successful cleanup.")
    lines.extend(["", "## Actual model activity", ""])
    for case in report["cases"]:
        for event in case["model_calls"]:
            lines.append(f"- {case['id']}: {event['model']}, {event['status']}, {event['elapsed_seconds']:.3f}s.")
    for case in completed:
        semantic = case.get("semantic_report") or {}
        if semantic.get("status") == "not_applicable":
            lines.append(f"- {case['id']}: semantic comparison not applicable — {semantic.get('coverage_reason', 'see case report')}.")
        for comparison in semantic.get("comparisons", []):
            if comparison.get("editorial"):
                decision = comparison["editorial"]
                probability = comparison.get("later_covers_earlier", {}).get("entailment")
                lines.append(f"- {case['id']}: Qwen chose '{decision.get('keep')}'; later-covers-earlier NLI entailment={probability}. {comparison.get('blocked', '')}")
    quiet = [f"{c['id']} ({issue['duration']:.2f}s)" for c in completed for issue in c["qa"]["issues"] if issue["kind"] == "long_quiet_span"]
    audio_flags = [issue for c in completed for issue in c["qa"]["issues"] if issue["kind"] in ("clipping", "join_transient")]
    lines.extend(["", "## Render inspection", "",
        f"{sum(c['qa']['geometry']=='PASS' for c in completed)}/{len(completed)} renders passed decoded picture/audio timestamp geometry checks. Source-selection text and a fresh rendered-audio ASR were compared.",
        "Quiet-span flags: " + (", ".join(quiet) or "none") + ". Protected reveal timing is intentional.",
        f"Clipping/abrupt-join transient flags: {len(audio_flags)}. These measurements do not establish perceptual quality.", "",
        "Plans, per-case reports, rendered MP4s, mapped caption sidecars and independent ASR artifacts are stored in the case subdirectories. `fixture-integrated-comparison-snapshot.json` preserves the exact comparison results used by this run.", "",
        "Naturalness, audible consonant damage, perceptual lip sync, and human preference remain unverified. These are synthetic Windows speech recordings, not representative natural stuttering or user footage."])
    (Path(run) / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="Synthetic fixture IDs; default all nine")
    args = parser.parse_args()
    folder = ROOT / "analysis" / "ai-evaluation"
    fixtures = json.loads((folder / "fixtures.json").read_text(encoding="utf-8"))
    if fixtures.get("synthetic") is not True:
        raise ValueError("This evaluation accepts explicitly synthetic fixture media only.")
    run = folder / ("local-baseline-" + str(time.time_ns()))
    run.mkdir(parents=True, exist_ok=False)
    comparison_path = folder / "evaluation-results.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {}
    other_cases = {row["id"]: row for row in comparison.get("cases", [])}
    save_json(run / "fixture-integrated-comparison-snapshot.json", comparison)
    report = dict(scope="Actual current automatic_cut.plan, actual local models when eligible, actual review renders and independent ASR",
        synthetic=True, openai_calls=0, actual_api_cost_usd=0.,
        fixture_comparison_mode=comparison.get("mode"),
        comparison_warning="The other pipeline uses deterministic fixture responses; this is not a live OpenAI model-quality comparison.",
        code_sha256_before=code_identity(), cases=[],
        listening="NOT VERIFIED", human_preference="NOT MEASURED")
    save_json(run / "report.json", report)
    print("REPORT_DIRECTORY " + str(run), flush=True)
    all_started = time.monotonic()
    render_model = None
    for case in fixtures["cases"]:
        if args.cases and case["id"] not in args.cases:
            continue
        case_dir = run / case["id"]
        case_dir.mkdir()
        logfile = case_dir / "progress.log"
        started = time.monotonic()
        events = []
        entry = dict(id=case["id"], source=case["source"], source_seconds=case["duration"],
            expected=case["expected"], status="running", stage="analysis", model_calls=events,
            actual_api_cost_usd=0., listening="NOT VERIFIED", human_preference="NOT MEASURED")
        report["cases"].append(entry)

        def progress(message):
            line = f"[{case['id']}] +{time.monotonic()-started:.2f}s {message}"
            print(line, flush=True)
            with logfile.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

        try:
            progress("Starting complete local production analysis")
            analysis_started = time.monotonic()
            with measured_models(events, progress):
                result = plan(case["source"], 0, progress=progress, intensity="balanced",
                              protected_pauses=case.get("protected_pauses", []))
            entry["analysis_seconds"] = round(time.monotonic() - analysis_started, 4)
            entry["analysis_seconds_per_source_minute"] = round(entry["analysis_seconds"] / (case["duration"] / 60), 3)
            plan_path = case_dir / "local-plan.json"
            save_json(plan_path, result)
            transcript, timing_path = baseline_transcript(result)
            save_json(case_dir / "source-transcript.json", transcript)
            entry.update(plan=str(plan_path), source_timing_artifact=timing_path,
                kept=result["kept"], semantic_report=result["script_review"].get("semantic"),
                evidence_flags=result.get("evidence_flags", []), script_flags=result["script_review"].get("flags", []),
                number_of_cuts=len(complement(result["kept"], 0, result["duration"])))
            retained = [word for segment in transcript for word in segment["words"]
                        if any(a <= word["start"] and word["end"] <= b for a, b in result["kept"])]
            entry["source_text_metrics"] = text_metrics(case["expected"], " ".join(w["text"].strip() for w in retained))
            output = case_dir / "local-after.mp4"
            entry["stage"] = "render"
            render_started = time.monotonic()
            original_words = [w for s in transcript for w in s["words"]]
            rendering = render_preview(case["source"], result["kept"], result["duration"], result["fps"],
                                       output, progress=progress, annotations=original_words)
            entry["render_seconds"] = round(time.monotonic() - render_started, 4)
            entry["output_seconds"] = rendering["frame_map"][-1]["output_end_frame"] / result["fps"]
            entry["output"] = str(output)
            expected_words = []
            for word in original_words:
                for span in map_source_span(rendering["frame_map"], word["start"], word["end"]):
                    expected_words.append(dict(word, start=span["output_start"], end=span["output_end"], partial=span["partial"]))
            if render_model is None:
                from faster_whisper import WhisperModel
                render_model = WhisperModel(str(ROOT / ".model-cache" / "speech-small.en"),
                                            device="cpu", compute_type="int8", cpu_threads=4)
            qa_started = time.monotonic()
            entry["stage"] = "independent_render_asr"
            observed = transcribe_render(output, 0, entry["output_seconds"], progress=progress, model=render_model)
            entry["rendered_text_metrics"] = text_metrics(case["expected"],
                " ".join(w["text"].strip() for w in observed["words"]))
            entry["stage"] = "render_geometry_audio_audit"
            qa = audit_render(output, rendering["frame_map"], result["fps"], microphone=0,
                              expected_words=expected_words, observed_words=observed["words"], progress=progress)
            entry["qa_seconds"] = round(time.monotonic() - qa_started, 4)
            entry["qa"] = qa
            semantic = entry["semantic_report"] or {}
            entry["status"] = "completed_with_incomplete_semantic_review" if semantic.get("status") == "incomplete" else "completed"
            entry["stage"] = "complete"
            if case["id"] in other_cases:
                other = other_cases[case["id"]]["outputs"]["ai"]
                entry["fixture_integrated_comparison"] = dict(
                    output=other["output"], output_seconds=other["duration"],
                    exact_expected_token_match=other["exact_expected_token_match"],
                    retained_source_text=other["retained_source_text"], qa=other["qa"],
                    local_minus_fixture_output_seconds=round(entry["output_seconds"] - other["duration"], 4))
            progress(f"Completed: local {entry['output_seconds']:.3f}s, geometry {qa['geometry']}, word check {qa['word_check']['status']}")
        except Exception as error:
            entry.setdefault("analysis_seconds", round(time.monotonic() - analysis_started, 4))
            entry.update(status="failed", error_type=type(error).__name__, error=str(error))
            progress(f"FAILED {type(error).__name__}: {error}")
        finally:
            entry["total_seconds"] = round(time.monotonic() - started, 4)
            save_json(case_dir / "case-report.json", entry)
            report["total_seconds"] = round(time.monotonic() - all_started, 4)
            report["code_sha256_after"] = code_identity()
            report["code_changed_during_run"] = report["code_sha256_before"] != report["code_sha256_after"]
            save_json(run / "report.json", report)
    report["complete"] = len(report["cases"]) == len(args.cases or fixtures["cases"])
    report["failed_cases"] = [case["id"] for case in report["cases"] if case["status"] == "failed"]
    report["qwen_calls"] = sum(event["model"] == "Qwen3-4B-Q4_K_M" for case in report["cases"] for event in case["model_calls"])
    report["incomplete_qwen_calls"] = sum(event["model"] == "Qwen3-4B-Q4_K_M" and event["status"] != "completed"
                                          for case in report["cases"] for event in case["model_calls"])
    save_json(run / "report.json", report)
    write_summary(run, report)
    print("REPORT " + str(run / "report.json"), flush=True)
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
