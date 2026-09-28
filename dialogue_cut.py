"""Tight word boundaries with restart, abandoned-take and repeat removal."""
import re
from difflib import SequenceMatcher
from review_core import merge_intervals,complement
from dialogue_flow import disfluencies,speech_ranges
from speech_safety import facts,compatible_words,quoted_or_demonstrated,source_words

def token(w):return re.sub('[^a-z0-9]','',w['text'].lower())

def canonical(ts):
    noise={'so','alright','guys','honestly','oh'}
    out=[]
    for t in ts:
        if t=='everythings':out.extend(['everything','is'])
        else:out.append(t)
    while out and out[0] in noise:out.pop(0)
    if out[-3:]==['to','be','honest']:out=out[:-3]
    return out

CONTENT_STOP={'a','an','the','i','im','you','he','she','it','we','they','is','are','was','were','be','been',
              'do','does','did','have','has','had','of','to','for','in','on','at','with','from','and','or',
              'but','so','that','this','there','here','my','your','our','their',
              'think','will','would','should','could','can','gonna','going'}
RESTART_STEMS=(('i','think'),('i','dont'),('i','just'),('im','gonna'),('im','going','to'),
               ('we','have'),('we','need'),('were','gonna'),('were','going','to'),
               ('this','is'),('thats','why'),('if','you'),('as','you','can','see'))

def common_prefix(a,b):
    n=0
    while n<min(len(a),len(b)) and a[n]==b[n]:n+=1
    return n

def content(ts):return {t for t in ts if t and t not in CONTENT_STOP and len(t)>1}

def unfinished(words):
    text=words[-1]['text'].rstrip()
    ts=[token(w) for w in words]
    return text.endswith(('...', '…', '-', '—')) or ts[-1] in {'a','an','the','to','of','with','because','if','and','but'}

def related_restart(a,b,distance):
    """Only lexical identity authorizes a rule-based complete-take deletion."""
    if distance>12:return False
    ca,cb=canonical(a),canonical(b)
    if not ca or not cb:return False
    if ca==cb and tuple(ca) in {('my','god'),('no','way')}:return False
    return ca==cb and len(ca)>=4 and len(content(ca))>=2

def clean_dialogue(transcript,duration):
    words=[w for w in source_words(transcript) if 0<=w['start']<=w['end']<=duration+.1]
    words,disfluent=disfluencies(words,duration)
    groups=[]
    for w in words:
        if not groups or w['start']-groups[-1][-1]['end']>.65 or (token(w)=='so' and len(groups[-1])>=5 and w['start']-groups[-1][-1]['end']>.3) or (len(groups[-1])>=4 and groups[-1][-1]['text'].rstrip().endswith(('.', '?', '!'))):groups.append([])
        groups[-1].append(w)
    split=[]
    for group in groups:
        ts=[token(w) for w in group];point=None
        if len(set(ts[:4]))>=3:
            for j in range(5,len(ts)-3):
                if ts[:4]==ts[j:j+4] and group[j]['start']-group[0]['start']<=20:
                    point=j-1 if ts[j-1]=='alright' else j;break
        split.extend([group[:point],group[point:]] if point else [group])
    groups=split;removed=[]
    # An isolated grammatical lead-in with no payload is an abandoned start.
    # Do not classify a whole sentence as unfinished merely from ASR punctuation.
    empty_starts={('i','was'),
                  ('i','was','gonna'),('im','gonna'),('im','going','to'),('we','were','gonna')}
    for i,group in enumerate(groups[:-1]):
        ts=tuple(token(w) for w in group)
        next_group=groups[i+1]
        gap=next_group[0]['start']-group[-1]['end']
        if ts in empty_starts and 1.2<=gap<=12 and compatible_words(group,next_group):
            removed.append(dict(start=max(0,group[0]['start']-.075),end=group[-1]['end']+.12,
                earlier=''.join(w['text'] for w in group).strip(),kept=''.join(w['text'] for w in next_group).strip(),
                kept_start=next_group[0]['start'],reason='Abandoned grammatical start before a long pause'))
    for i,old in enumerate(groups):
        a=[token(w) for w in old]
        if len(a)<2 or len(set(a))<2:continue
        if quoted_or_demonstrated(''.join(w['text'] for w in old)):continue
        if old[-1]['text'].rstrip().endswith(('!','?')):continue
        for new in groups[i+1:]:
            distance=new[0]['start']-old[-1]['end']
            if distance>180:break
            b=[token(w) for w in new]
            if len(b)<2:continue
            if not compatible_words(old,new) or quoted_or_demonstrated(''.join(w['text'] for w in new)):continue
            if sum(w.get('probability',1) for w in new)/len(new)<sum(w.get('probability',1) for w in old)/len(old)-.1:continue
            # Do not mistake contradictory statements or different numbers for retakes.
            if facts(''.join(w['text'] for w in old))!=facts(''.join(w['text'] for w in new)):continue
            ca,cb=canonical(a),canonical(b)
            # Prefer a finished take over a subsequent failed attempt. Later is
            # not automatically better, and ASR punctuation alone is weak evidence.
            if not unfinished(old) and (unfinished(new) or len(cb)<len(ca)*.65):continue
            if ca==cb and tuple(ca) in {('my','god'),('no','way')}:continue
            nearby=distance<=35
            same=nearby and related_restart(a,b,distance)
            prefix=nearby and unfinished(old) and len(ca)>=4 and len(cb)>=len(ca) and ca==cb[:len(ca)]
            if same or prefix:
                removed.append(dict(start=max(0,old[0]['start']-.09),end=min(duration,old[-1]['end']+.12),
                    earlier=''.join(w['text'] for w in old).strip(),kept=''.join(w['text'] for w in new).strip(),
                    kept_start=new[0]['start'],source_word_start=old[0]['start'],source_word_end=old[-1]['end'],
                    confidence=min(w.get('probability',1) for w in old+new),
                    reason='Exact nearby retake; complete replacement retained'))
                break
    # Restarts anywhere inside one breath, including after a valid lead-in such as
    # "I'm not gonna lie ...". Keep that lead-in and remove only the abandoned clause.
    for group in groups:
        if not compatible_words(group) or quoted_or_demonstrated(''.join(w['text'] for w in group)):continue
        ts=[token(w) for w in group]
        found=None
        for j in range(3,len(ts)-1):
            for start in range(max(0,j-14),j-2):
                first=canonical(ts[start:j]);later=canonical(ts[j:])
                if len(first)<2 or len(later)<2:continue
                prefix=common_prefix(first,later)
                stem=any(tuple(first[:len(s)])==s and tuple(later[:len(s)])==s for s in RESTART_STEMS)
                if prefix<3 and not stem:continue
                # A common grammatical stem is not evidence that a changed
                # claim, qualification, or corrected quantity is expendable.
                if facts(''.join(w['text'] for w in group[start:j]))!=facts(''.join(w['text'] for w in group[j:])):continue
                if first!=later[:len(first)] or len(content(first))<2:continue
                if any(w['text'].rstrip().endswith(('.', '!', '?')) for w in group[start:j]):continue
                if group[j]['start']-group[start]['start']>20:continue
                found=(start,j);break
            if found:break
        if found:
            start,j=found
            removed.append(dict(start=max(0,group[start]['start']-.09),end=max(0,group[j]['start']-.09),
                earlier=''.join(w['text'] for w in group[start:j]).strip(),kept=''.join(w['text'] for w in group[j:]).strip(),
                kept_start=group[j]['start'],source_word_start=group[start]['start'],source_word_end=group[j-1]['end'],
                reason='Exact restarted phrase; later version kept'))
    removed.extend(disfluent)
    # Retake padding must not truncate an adjacent retained word. Clamp each edge
    # against real word boundaries, then deduplicate overlapping detector evidence.
    for row in removed:
        limit=row.get('cut_limit',max(row.get('source_word_end',0),row['kept_start']-.075))
        row['end']=min(row['end'],limit,row['kept_start'])
        if 'source_word_start' in row:
            row['start']=max(row['start'],max((w['end'] for w in words if w['end']<=row['source_word_start']),default=0))
    removed.sort(key=lambda r:(r['start'],-r['end']))
    unique=[]
    for row in removed:
        if row['end']<=row['start']:continue
        if any(x['start']<=row['start'] and x['end']>=row['end'] for x in unique):continue
        unique.append(row)
    cuts=merge_intervals([(r['start'],r['end']) for r in unique],0,duration)
    return subtract(speech_ranges(words,duration),cuts),cuts,unique

def subtract(ranges,cuts):
    return [r for a,b in ranges for r in complement(cuts,a,b)]
