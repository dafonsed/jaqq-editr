"""Sentence-level retake review with directional, local entailment checks.

Similarity retrieves candidates; it is never sufficient to authorize deletion.
Only remove a span when a surviving take covers its meaning. Source coordinates
are retained throughout, including when a sentence crosses multiple edit clips.
"""
import json
import re
from functools import lru_cache

from semantic_review import encode, facts, opposite, terms
from app_paths import ROOT
import numpy as np
from speech_safety import source_words,compatible_words,quoted_or_demonstrated

VERSION = 'context-retakes-2'


def protected_facts(text):
    # "one of the most trusted" is a superlative, not a quantity correction.
    text = re.sub(r'\bone of (?:the |our )?', '', text, flags=re.I)
    return facts(text)


def model_text(text):
    # Strip conversational lead-ins, not the proposition or its negation.
    return re.sub(r'^(?:(?:and|so|alright|guys)[, ]+)+', '', text.strip(), flags=re.I)


@lru_cache(maxsize=1)
def runtime():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    folder = ROOT / '.model-cache/retake-nli'
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder / 'onnx/model.onnx'),
        sess_options=options, providers=['CPUExecutionProvider'])
    tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    tokenizer.enable_truncation(max_length=384)
    labels = json.loads((folder / 'config.json').read_text())['id2label']
    return session, tokenizer, labels


def entailment(pairs, cancel=lambda: False):
    session, tokenizer, labels = runtime()
    result = []
    # Pair-at-a-time bounds memory and gives a cancellation point per inference.
    for start in range(0, len(pairs)):
        if cancel():
            from automatic_cut import Cancelled
            raise Cancelled()
        batch = tokenizer.encode_batch(pairs[start:start + 1])
        ids = np.asarray([x.ids for x in batch], dtype=np.int64)
        inputs = dict(input_ids=ids,
            attention_mask=np.asarray([x.attention_mask for x in batch], dtype=np.int64),
            token_type_ids=np.asarray([x.type_ids for x in batch], dtype=np.int64))
        logits = session.run(None, {x.name: inputs[x.name] for x in session.get_inputs()})[0]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        result.extend({labels[str(i)]: float(p) for i, p in enumerate(row)} for row in probabilities)
    return result


def sentence_spans(transcript, ranges):
    words = source_words(transcript)
    groups = []
    previous_index = None
    for index, word in enumerate(words):
        # A partly selected word is not enough evidence for an automatic deletion.
        if not any(a <= word['start'] + .035 and b >= word['end'] - .035 for a, b in ranges):
            continue
        split = not groups or previous_index != index - 1
        if groups and not split:
            previous = groups[-1][-1]
            split = (word['start'] - previous['end'] > 1.2 or
                word.get('speaker') != previous.get('speaker') or
                previous['text'].rstrip().endswith(('.', '?', '!', '…', '—', '-')) or
                len(groups[-1]) >= 60)
        if split:
            groups.append([])
        groups[-1].append(word)
        previous_index = index
    spans = []
    for index, group in enumerate(groups):
        text = ''.join(w['text'] for w in group).strip()
        if len(terms(text)) < 5:
            continue
        spans.append(dict(id=index, start=group[0]['start'], end=group[-1]['end'],
            text=text, words=group, complete=(not text.endswith(('-', '—', '…', '...')) and
                terms(text)[-1] not in {'and','but','to','the','a','because','if','with','of'}),
            confidence=sum(w.get('probability', 1) for w in group) / len(group)))
    return spans, words


def review(transcript, ranges, events=(), progress=lambda _: None, cancel=lambda: False, editorial=False):
    spans, words = sentence_spans(transcript, ranges)
    report = dict(version=VERSION, model='nli-deberta-v3-small float32',
        comparisons=[], changes=[], sentences=len(spans),status='completed',
        scope='Nearby alternative takes; global story structure and visual meaning require review',
        nli_pairs_checked=0,editor_pairs_checked=0,warnings=[],
        embedding_status='not_run',nli_status='not_run',
        editor_status='not_run' if editorial else 'not_requested')
    if len(spans) < 2:
        report['status']='not_applicable'
        report['coverage_reason']='Fewer than two eligible sentence spans; no semantic comparison was performed'
        return ranges, report
    embeddings = encode([s['text'] for s in spans])
    report['embedding_status']='completed'
    preference_file = ROOT / 'semantic-preferences.json'
    preferences = json.loads(preference_file.read_text(encoding='utf-8')).get('approved_alternative_takes', []) if preference_file.is_file() else []
    candidates = []
    for i, a in enumerate(spans):
        for j in range(i + 1, len(spans)):
            b = spans[j]
            gap = b['start'] - a['end']
            if gap > 60:
                break
            score = float(embeddings[i] @ embeddings[j])
            approved = any(''.join(terms(a['text'])) == ''.join(terms(p['earlier'])) and ''.join(terms(b['text'])) == ''.join(terms(p['keep'])) for p in preferences)
            if score < .40 and gap > 15 and not approved:
                continue
            blocked = None
            if protected_facts(a['text']) != protected_facts(b['text']):
                blocked = 'Different number or negation; preserve both'
            elif opposite(a['text'], b['text']):
                blocked = 'Opposing claims; preserve both'
            elif any(a['start'] < y and b['end'] > x for x, y in events):
                blocked = 'Action context; preserve both'
            elif any(t in terms(b['text'])[:3] for t in ('also', 'another', 'however', 'additionally')):
                blocked = 'Explicit additional point; preserve both'
            elif '?' in a['text'] or '?' in b['text']:
                blocked = 'Question or response; preserve both'
            elif min(a['confidence'], b['confidence']) < .60:
                blocked = 'Uncertain transcription; preserve both'
            elif not compatible_words(a['words'],b['words']):
                blocked = 'Different speakers or uncertain word alignment; preserve both'
            elif quoted_or_demonstrated(a['text']) or quoted_or_demonstrated(b['text']):
                blocked = 'Quoted or demonstrated speech; preserve both'
            entry = dict(first=a['text'], second=b['text'], first_start=a['start'],
                second_start=b['start'], similarity=score, blocked=blocked, user_approved=approved)
            report['comparisons'].append(entry)
            if not blocked:
                candidates.append((i, j, entry))
    if not candidates:
        report['coverage_reason']='No eligible alternative-take pairs passed the retrieval and meaning safeguards'
        report['nli_status']='not_applicable'
        if editorial:report['editor_status']='not_applicable'
        return ranges, report
    progress(f'Checking meaning and alternate takes: {len(candidates)} sentence pairs…')
    pairs = []
    for i, j, _ in candidates:
        a,b=model_text(spans[i]['text']),model_text(spans[j]['text'])
        pairs.extend([(b,a),(a,b)])
    scores = entailment(pairs, cancel)
    if len(scores)!=len(pairs) or any(not {'entailment','contradiction','neutral'}<=set(s) for s in scores):
        raise ValueError('Incomplete meaning-check results; no semantic deletions applied')
    report['nli_pairs_checked']=len(candidates)
    report['nli_status']='completed'
    editorial_decisions={}
    if editorial:
        from local_editor import Editor
        # Only inspect plausible same-topic candidates, never arbitrary pairs.
        pending=[(n,i,j) for n,(i,j,entry) in enumerate(candidates)
                 if entry['similarity']>=.30 and not entry['user_approved']]
        if pending:
            progress('Reading alternate takes with the on-device editor…')
            with Editor(cancel) as editor:
                for index,(n,i,j) in enumerate(pending):
                    progress(f'Reading alternate takes: {index+1} / {len(pending)}…')
                    a,b=spans[i],spans[j]
                    surrounding=dict(
                        before=''.join(w['text'] for w in words if a['start']-12<=w['start']<a['start']).strip(),
                        between=''.join(w['text'] for w in words if a['end']<=w['start']<b['start']).strip(),
                        after=''.join(w['text'] for w in words if b['end']<=w['start']<=b['end']+12).strip())
                    editorial_decisions[n]=editor.verify(a['text'],b['text'],context=surrounding)
        report['editorial_model']='Qwen3-4B-Q4_K_M; proposal plus information-loss audit'
        report['editor_pairs_checked']=len(editorial_decisions)
        report['editor_status']='completed' if pending else 'not_applicable'
        if any(d.get('status')=='incomplete' for d in editorial_decisions.values()):
            report['status']=report['editor_status']='incomplete'
            report['warnings'].append('Incomplete editorial response; affected speech was preserved for review')
    edges = []
    for n, (i, j, entry) in enumerate(candidates):
        later_covers, earlier_covers = scores[2*n:2*n+2]
        entry.update(later_covers_earlier=later_covers, earlier_covers_later=earlier_covers)
        decision=editorial_decisions.get(n)
        if decision:entry['editorial']=decision
        # Direction matters: a short summary cannot replace an informative take.
        forward = later_covers['entailment'] >= .92 and later_covers['contradiction'] <= .03 and spans[j]['complete']
        reverse = earlier_covers['entailment'] >= .92 and earlier_covers['contradiction'] <= .03 and spans[i]['complete']
        if entry['user_approved']:
            # A saved text preference may have come from another recording or
            # older settings. It can rank candidates, never replace current
            # evidence that the surviving take covers this source span.
            if not forward:
                entry['blocked']='Saved preference lacks current directional meaning coverage; preserve both'
                continue
            loser, winner = i, j
        elif decision:
            selected=decision['keep']
            covered=later_covers if selected=='B' else earlier_covers
            winner=j if selected=='B' else i
            if selected not in {'A','B'} or not (forward if selected=='B' else reverse):
                entry['blocked']='Editorial review did not confirm a safe replacement'
                continue
            loser=i if winner==j else j
        elif forward and reverse:
            winner = j if spans[j]['confidence'] >= spans[i]['confidence'] - .1 else i
            loser = i if winner == j else j
        elif forward:
            loser, winner = i, j
        elif reverse:
            loser, winner = j, i
        else:
            entry['blocked'] = 'Meaning is not confidently covered; preserve both'
            continue
        confidence = (later_covers if winner == j else earlier_covers)['entailment']
        edges.append((confidence, loser, winner, entry))
    # Never remove both ends of a chain. Every deletion has a direct, surviving
    # replacement, not an assumption that semantic similarity is transitive.
    removed, protected = set(), set()
    cuts = []
    for confidence, loser, winner, entry in sorted(edges, key=lambda x: (not x[3]['user_approved'],-x[0])):
        if loser in removed or loser in protected or winner in removed:
            entry['blocked'] = 'Conflicting take chain; preserve surviving replacement'
            continue
        a, b = spans[loser], spans[winner]
        before = max((w['end'] for w in words if w['end'] <= a['start'] and w is not a['words'][0]), default=0)
        after = min((w['start'] for w in words if w['start'] >= a['end'] and w is not a['words'][-1]), default=a['end'] + .12)
        start, end = max(before, a['start'] - .075), min(after, a['end'] + .12)
        # Consume the outer handles when the entire selected clip is this take.
        for x, y in ranges:
            if x <= a['start'] <= y and not any(x <= w['start'] < a['start'] for w in words):
                start = x
            if x <= a['end'] <= y and not any(a['end'] < w['end'] <= y for w in words):
                end = y
        if start >= end:
            continue
        removed.add(loser)
        protected.add(winner)
        cuts.append((start, end))
        entry['decision'] = 'Remove first' if loser < winner else 'Remove second'
        report['changes'].append(dict(source_start=start, source_end=end, text=a['text'],
            kept_text=b['text'], kept_source_start=b['start'], entailment=confidence,
            reason='User-confirmed alternative introduction' if entry['user_approved'] else
                'On-device editor and information-loss audit confirmed the replacement' if entry.get('editorial') else
                'Nearby alternative sentence; surviving take covers its meaning'))
    from dialogue_cut import subtract
    result = subtract(ranges, cuts)
    result = [(a, b) for a, b in result if any(a < w['end'] and b > w['start'] for w in words)
              or any(a < y and b > x for x, y in events)]
    return result, report
