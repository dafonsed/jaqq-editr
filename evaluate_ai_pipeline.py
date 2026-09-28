"""Render before/after fixtures and report measured evidence, without overclaims.

Default is a deterministic decision adapter using fixture expectations, NOT live
model evaluation. --live uses real OpenAI calls and requires OPENAI_API_KEY.
The legacy comparison explicitly covers existing deterministic stages only when
the full local Qwen reviewer is unavailable. Human preference stays unmeasured.
"""
import argparse
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import time

from app_paths import ROOT
from automatic_cut import extract, key_for, select_ranges
from ai_pipeline import plan_ai, _local_timing, validate_for_export, write_ai_artifacts
from pause_cleanup import pause_settings, clean_pauses, bridge_word_gaps
from cut_integrity import protect_words
from review_core import build_frame_map, save_json
from review_render import render_preview, audit_render
from speech_edges import refine_edges


def normalized(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace('’', "'"))


class FixtureDecisions:
    """Expected-output replay exercises integration, never model intelligence."""
    def __init__(self, case):
        self.case = case
        self.ledger = []
        self.claimed = set()

    def transcribe(self, wav, **kwargs):
        folder = ROOT/'analysis'/'openai'/f"{key_for(self.case['source'])}-mic0"
        paths = sorted(folder.glob('local-timing-*.json'))
        local = json.loads(paths[-1].read_text(encoding='utf-8'))
        # Local timing is real inference; cloud text is explicitly replayed here.
        return dict(text=' '.join(w['text'] for s in local for w in s['words']),
                    model='fixture:gpt-transcribe', api_call_occurred=False, cached=False,
                    request_id=None, provenance='fixture')

    def responses(self, model, payload, schema, **kwargs):
        window = json.loads(payload['input'])
        words = window['words']
        lookup = {w['source_word_id']: w for w in words}
        tokens = [normalized(w['text']) for w in words]
        flat = [t for part in tokens for t in part]
        target = normalized(self.case['expected'])
        deleted = set()
        offsets = [i for i, part in enumerate(tokens) for _ in part]
        for op, a, b, x, y in SequenceMatcher(None, flat, target, autojunk=False).get_opcodes():
            if op == 'delete':
                deleted.update(words[offsets[i]]['source_word_id'] for i in range(a,b))
        candidates = window['candidates']
        ranked = sorted(candidates, key=lambda c: (
            {'correction':0,'retake':1,'repetition':2,'repair':3,'filler':4,'pause':5}.get(c['kind'],6),
            -len(c['word_ids'])))
        rows = {}
        for candidate in ranked:
            chosen = [x for x in candidate['word_ids'] if x in deleted]
            action = 'KEEP'
            if candidate['kind'] == 'pause' and not self.case.get('protect_gap'):
                action = 'SHORTEN_PAUSE'
            elif chosen and not set(chosen) & self.claimed:
                indexes = [next(i for i,w in enumerate(words) if w['source_word_id']==x) for x in chosen]
                if indexes == list(range(indexes[0],indexes[-1]+1)):
                    action = 'REPLACE_EARLIER_TAKE_WITH_LATER_TAKE' if candidate['kind']=='retake' else 'REMOVE'
                    self.claimed.update(chosen)
            if action == 'KEEP':
                chosen = []
            rows[candidate['id']] = dict(candidate_id=candidate['id'], action=action,
                remove_word_ids=chosen,
                keep_word_ids=[x for x in candidate['review_word_ids'] if x not in chosen],
                source_start=lookup[chosen[0]]['start'] if chosen else candidate['start'],
                source_end=lookup[chosen[-1]]['end'] if chosen else candidate['end'],
                reason='Deterministic expected-output fixture replay; not an AI semantic judgment.',
                confidence=.99, required_context_word_ids=[], meaning_change=False,
                ambiguous=False, needs_audio_review=False)
        return dict(data={'decisions':[rows[c['id']] for c in candidates]},
                    model='fixture:'+model, api_call_occurred=False, cached=False,
                    provenance='fixture', request_id=None)

    def summary(self):
        return dict(provenance='fixture', actual_api_calls=0, cost_usd_estimate=0.,
                    live_cost_measurement='NOT MEASURED', pricing_is_estimate=False)


def legacy_stages(case, transcript, wav, fps):
    started = time.monotonic()
    duration = case['duration']
    protected = [(p['start'],p['end']) for p in case.get('protected_pauses', [])]
    kept, speech, duplicates = select_ranges(transcript, [], duration, protected)
    exclusions = [(r['start'],r['end']) for r in duplicates]
    kept, _ = refine_edges(kept, wav, protected, exclusions, transcript=transcript)
    kept, _ = bridge_word_gaps(kept, transcript, exclusions=exclusions, protected_pauses=protected)
    kept, _ = clean_pauses(kept, wav, protected, transcript=transcript,
        settings=pause_settings('balanced'), exclusions=exclusions,
        protected_pauses=case.get('protected_pauses', []))
    kept, _ = protect_words(kept, transcript, exclusions, duration, fps)
    return kept, round(time.monotonic()-started,3)


def text_at(transcript, kept):
    return ' '.join(w['text'].strip() for s in transcript for w in s['words']
                    if any(a<=w['start'] and w['end']<=b for a,b in kept))


def text_metrics(expected, actual):
    reference, observed = normalized(expected), normalized(actual)
    return dict(expected=expected, retained_source_text=actual,
        exact_expected_token_match=reference==observed,
        differences=[dict(operation=op,expected=reference[a:b],actual=observed[x:y])
                     for op,a,b,x,y in SequenceMatcher(None,reference,observed,autojunk=False).get_opcodes()
                     if op!='equal'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Use billable OpenAI calls, with the configured server-side API key')
    parser.add_argument('--cases', nargs='*', help='Fixture IDs; defaults to all')
    args = parser.parse_args()
    folder = ROOT/'analysis'/'ai-evaluation'
    rendered = folder / ('run-'+str(time.time_ns()))
    rendered.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads((folder/'fixtures.json').read_text(encoding='utf-8'))['cases']
    results = []
    for case in fixtures:
        if args.cases and case['id'] not in args.cases:
            continue
        print('Evaluating',case['id'],flush=True)
        start = time.monotonic()
        result = plan_ai(case['source'],0,progress=lambda s: print(s,flush=True),
            protected_pauses=case.get('protected_pauses',[]),
            client=None if args.live else FixtureDecisions(case))
        result['evaluation_provenance'] = 'live_openai' if args.live else 'fixture_decisions_no_live_API'
        if not args.live:
            result['ai_plan']['transcription']['evaluation_provenance']='local_ASR_replay_not_gpt-transcribe'
        validate_for_export(result)
        save_json(folder/(case['id']+'-ai-plan.json'),result)
        legacy, old_seconds = legacy_stages(case,result['transcript'],result['analysis_audio'],result['fps'])
        outputs = {}
        for label, kept in [('old-stages',legacy),('ai',result['kept'])]:
            path = rendered/(case['id']+'-'+label+'.mp4')
            render_started = time.monotonic()
            render = render_preview(case['source'],kept,result['duration'],result['fps'],path)
            qa = audit_render(path,render['frame_map'],result['fps'],microphone=0)
            rendered_seconds = time.monotonic()-render_started
            outputs[label] = dict(kept=kept,render_seconds=round(rendered_seconds,3),
                duration=render['frame_map'][-1]['output_end_frame']/result['fps'],
                output=str(path),qa=qa,**text_metrics(case['expected'],text_at(result['transcript'],kept)))
        write_ai_artifacts(folder/(case['id']+'-artifacts'),result)
        results.append(dict(id=case['id'],category=case['category'],source=case['source'],
            source_seconds=result['duration'],outputs=outputs,metrics=result['ai_plan']['metrics'],
            api_execution=dict(live_requested=args.live,
                transcription_status=result['ai_plan']['transcription'].get('status'),
                decision_warnings=result['ai_plan'].get('warnings',[]),
                calls=result['ai_plan'].get('calls',[])),
            plan=str(folder/(case['id']+'-ai-plan.json')),old_decision_stage_seconds=old_seconds,
            total_evaluation_seconds=round(time.monotonic()-start,3),
            listening='NOT VERIFIED',human_preference='NOT MEASURED'))
        save_json(folder/'evaluation-results.json',dict(
            mode='live_openai' if args.live else 'fixture_decisions_no_live_API',
            scope='Synthetic speech + actual extraction/local word timing/validation/rendering. Fixture mode measures integration, NOT OpenAI model quality.',
            baseline_scope='Existing deterministic speech/edge/pause stages. Full Qwen semantic reviewer was not included.',
            cases=results, live_requested=args.live, live_api_verified=False,
            verification_note='Invocation mode alone does not prove successful model calls or editing quality; inspect api_execution and independently review renders.',
            human_preference='NOT MEASURED',natural_speech_quality='NOT VERIFIED'))
        print('Rendered',case['id'],outputs['old-stages']['duration'],'->',outputs['ai']['duration'],flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
