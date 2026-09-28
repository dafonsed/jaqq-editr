"""Word-level disfluencies and phrase-aware handles, before picture selection."""
import re
from speech_safety import (NUMBERS, compatible_words, quoted_or_demonstrated,
                           sentence_context, facts)

LEAD=.075
TAIL=.12
SENTENCE_TAIL=.24
INTERNAL_PAUSE=.48
FILLERS={'um','uh','erm','er'}
FUNCTION={'i','im','we','were','the','a','to','it','its','and','but','so','this','that','you','because','if','when','then','they','he','she'}

def norm(w):return re.sub('[^a-z0-9]','',w['text'].lower())

def disfluencies(words,duration):
    removed=[];dead=set()
    def discard(a,b,reason):
        if not compatible_words(words[a:min(len(words),b+4)]):return False
        if quoted_or_demonstrated(sentence_context(words,a)):return False
        # Remove only the earlier attempt; never consume the next spoken onset.
        left=max(0,words[a]['start']-LEAD)
        # A lead handle must not preserve the tail of the discarded syllable.
        # Abutting words have no room for a handle; overlapping alignment is
        # ambiguous, so leave that attempt intact instead of clipping either word.
        if words[b-1]['end']>words[b]['start']:return False
        right=min(words[b]['start'],max(words[b]['start']-LEAD,words[b-1]['end']))
        if a and words[a-1]['end']>left:left=words[a-1]['end']
        if right<=left:return False
        dead.update(range(a,b))
        removed.append(dict(start=left,end=right,earlier=''.join(w['text'] for w in words[a:b]).strip(),
            kept=''.join(w['text'] for w in words[b:b+8]).strip(),kept_start=words[b]['start'],cut_limit=right,
            confidence=min(w.get('probability',1) for w in words[a:min(len(words),b+4)]),reason=reason))
        return True
    # Explicit self-repair is directional evidence, unlike mere similarity.
    # Only splice a quantity into its existing grammatical slot, or retain an
    # independently spoken replacement clause. Other corrections stay reviewable.
    subjects={'i','im','we','you','he','she','it','its','they','this','that','what','dont','do'}
    for marker in range(1,len(words)-1):
        if norm(words[marker]) not in {'sorry','actually'}:continue
        before=words[marker-1]
        if words[marker+1]['start']-before['end']>2:continue
        keep=marker+1
        left=marker-1
        number=lambda w: norm(w) in NUMBERS or any(c.isdigit() for c in norm(w))
        replacement_number=(number(words[keep]) or (norm(words[keep]) in {'minus','negative','positive','plus'}
            and keep+1<len(words) and number(words[keep+1])))
        if number(before) and replacement_number:
            # Require an audible-repair marker plus a changed quantity.
            while left:
                previous=words[left-1]
                bridge=(norm(previous) in {'and','point'} and left>=2 and number(words[left-2]))
                sign=norm(previous) in {'minus','negative','positive','plus'}
                if not number(previous) and not bridge and not sign:break
                left-=1
            if facts(''.join(w['text'] for w in words[left:marker])) != facts(words[keep]['text']):
                discard(left,keep,'Explicit quantity correction; corrected amount retained')
        elif before['text'].rstrip().endswith(('-', '—', '…', '...')) and norm(words[keep]) in subjects:
            while left and marker-left<20:
                previous=words[left-1]
                if previous['text'].rstrip().endswith(('.', '!', '?')) or words[left]['start']-previous['end']>1.2:break
                left-=1
            # A preceding complete clause carries context the replacement may
            # not express. Keep it unless the repair starts a separate sentence.
            if not any(norm(w) in {'because','although','while','unless','if','when','before','after',
                                  'and','but','or','then'} for w in words[left:marker]):
                discard(left,keep,'Explicit clause correction; complete corrected instruction retained')
    for i,w in enumerate(words[:-1]):
        if i in dead:continue
        if quoted_or_demonstrated(sentence_context(words,i)):continue
        # Multiword cut-off starts must be removed as a span: "I wa— I wanted",
        # including successive source/transcription chunks.
        for j in range(i+2,min(i+10,len(words))):
            if any(k in dead for k in range(i,j+1)):break
            if words[j]['start']-words[j-1]['end']>1.2:break
            if not words[j-1]['text'].rstrip().endswith(('-', '—')):continue
            earlier=[norm(x) for x in words[i:j]]
            later=[norm(x) for x in words[j:j+len(earlier)]]
            partial=(len(later)==len(earlier) and earlier[:-1]==later[:-1] and
                     len(earlier[-1])>=2 and later[-1].startswith(earlier[-1]))
            scaffold={'what','you','need','to','i','we','was','were','want','wanted','going','gonna',
                      'mean','meant','trying','would','should','could','can','the','a','and'}
            abandoned=(all(t in scaffold for t in earlier) and earlier[-1] in {'to','the','a','and'}
                       and norm(words[j]) in subjects)
            if (partial or abandoned) and discard(i,j,'Abandoned opening replaced by its completed restart'):
                break
        if i in dead:continue
        # Restart with an intervening hesitation: "I, um, I think".
        j=i+1
        while j<len(words) and norm(words[j]) in FILLERS:j+=1
        if j>i+1 and j<len(words) and norm(w) in FUNCTION and norm(w)==norm(words[j]) and words[j]['start']-w['end']<1.2:
            discard(i,j,'Repeated word interrupted by a hesitation');continue
        # Long exact opening followed by an abandoned tail, then restarted.
        # Search across ASR punctuation groups, but never across a real pause.
        restarted=False
        for j in range(i+5,min(i+17,len(words))):
            if norm(w) not in {'i','im','we','were','not','this','that','because','if'}:break
            if any(words[k+1]['start']-words[k]['end']>.8 for k in range(i,j)):break
            n=0
            while i+n<j and j+n<len(words) and norm(words[i+n])==norm(words[j+n]):n+=1
            tail=words[i+n:j]
            if n<5 or len({norm(x) for x in words[i:i+n]})<4 or len(tail)>4:continue
            if any(x['text'].rstrip().endswith(('.','!','?')) for x in words[i:j]):continue
            if tail and norm(tail[-1]) not in {'this','that','and','but','the','a','to','im','i'}:continue
            # Include an optional "I'm" in the later retained lead-in.
            keep=j-1 if tail and norm(words[j-1])=='im' else j
            discard(i,keep,'Repeated long opening after an abandoned clause');restarted=True;break
        if restarted:continue
        # A recognizer may preserve an audible cut-off syllable as "th-".
        if w['text'].rstrip().endswith(('-', '—')) and len(norm(w))>=2 and norm(words[i+1]).startswith(norm(w)) and words[i+1]['start']-w['end']<.7:
            discard(i,i+1,'Cut-off word restarted immediately');continue
        if norm(w) in FILLERS and words[i+1]['start']-w['end']<1.2:
            discard(i,i+1,'Hesitation filler before continuing speech');continue
        # Longest repeated short prefix first; retain expressive single-word emphasis.
        for n in (8,7,6,5,4,3,2,1):
            if i+2*n>len(words):continue
            a=words[i:i+n];b=words[i+n:i+2*n]
            if any(j in dead for j in range(i,i+2*n)):continue
            if [norm(x) for x in a]!=[norm(x) for x in b]:continue
            if n==1 and norm(a[0]) in {'that','so'}:continue
            if norm(a[0]) not in FUNCTION and not (n>=3 and len({norm(x) for x in a})>=3):continue
            if any(x['text'].rstrip().endswith(('.', '!', '?')) for x in a):continue
            if norm(a[0])=='it' and i and norm(words[i-1]) not in FUNCTION:continue
            if any(y['start']-x['end']>.7 for x,y in zip(a+b,(a+b)[1:])):continue
            if n>1 and any(x['text'].rstrip().endswith(('!','?')) for x in a):continue
            discard(i,i+n,'Repeated word or short phrase stutter');break
    return [w for i,w in enumerate(words) if i not in dead],removed

def speech_ranges(words,duration):
    groups=[]
    for w in words:
        if not groups or w['start']-groups[-1][-1]['end']>INTERNAL_PAUSE:groups.append([])
        groups[-1].append(w)
    result=[]
    for i,g in enumerate(groups):
        tail=SENTENCE_TAIL if g[-1]['text'].rstrip().endswith(('.','!','?')) else TAIL
        end=min(duration,g[-1]['end']+tail)
        if i+1<len(groups):end=min(end,groups[i+1][0]['start']-LEAD)
        result.append([max(0,g[0]['start']-LEAD),end])
    return result
