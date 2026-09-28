"""Conservative microphone-energy refinement; never a semantic speech verdict."""
import math
import wave
import numpy as np
from review_core import complement, merge_intervals


def microphone_energy(wav, window=.01):
    """Read mono PCM without dropping the partial final window or its duration."""
    with wave.open(str(wav),'rb') as f:
        if f.getnchannels()!=1 or f.getsampwidth()!=2:
            raise ValueError('Boundary analysis requires mono 16-bit microphone audio')
        rate=f.getframerate()
        signal=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
    hop=max(1,round(rate*window));step=hop/rate
    duration=len(signal)/rate
    if not len(signal):return np.array([],dtype=np.float32),step,duration
    count=math.ceil(len(signal)/hop)
    padded=np.pad(signal,(0,count*hop-len(signal)))
    sizes=np.full(count,hop);sizes[-1]=len(signal)-(count-1)*hop
    energy=np.sqrt(np.sum(padded.reshape(-1,hop)**2,axis=1)/sizes)
    return energy,step,duration


def refine_edges(ranges,wav,events=(),exclusions=(),*,transcript=()):
    energy,step,duration=microphone_energy(wav)
    if not len(energy):return ranges,[]
    exclusions=merge_intervals(exclusions,0,duration)
    # Subtract even an interior barrier. Merely constraining outward extension
    # cannot enforce a rejected span that an earlier selection pass overlapped.
    ranges=[span for a,b in merge_intervals(ranges,0,duration)
            for span in complement(exclusions,a,b)]
    words=sorted((w for s in transcript for w in s.get('words',[])
                  if 0<w['end']-w['start']<3 and not any(
                      x-1e-6<=(w['start']+w['end'])/2<=y+1e-6 for x,y in exclusions)),
                 key=lambda w:w['start'])
    result=[];changes=[]
    for index,(a,b) in enumerate(ranges):
        lo=max(0,a-.3,result[-1][1] if result else 0)
        hi=min(duration,b+.3,ranges[index+1][0] if index+1<len(ranges) else duration)
        # Deleted takes are hard barriers, not silence to extend back across.
        for x,y in exclusions:
            if y<=a+.001:lo=max(lo,y)
            if x>=b-.001:hi=min(hi,x)
        left=max(0,int(lo/step));right=min(len(energy),int(hi/step))
        local=energy[left:right]
        if len(local)<3:result.append([a,b]);continue
        floor=float(np.percentile(local,15));peak=float(np.percentile(local,95))
        threshold=max(.0008,floor*3,peak*.045)
        if peak<threshold*2:result.append([a,b]);continue
        active=local>threshold
        # Ignore isolated clicks; require at least 30 ms of sustained energy.
        sustained=np.convolve(active.astype(int),np.ones(3,dtype=int),'valid')==3
        mask=np.zeros(len(local),dtype=bool)
        for offset in (0,1,2):mask[offset:offset+len(sustained)] |= sustained
        times=(np.flatnonzero(mask)+left)*step
        inside=times[(times>=a)&(times<b)]
        if not len(inside):result.append([a,b]);continue
        start=inside[0];end=inside[-1]+step
        # Follow only the same acoustic phrase across the proposed boundary.
        for t in times[times<start][::-1]:
            if start-(t+step)>.08:break
            start=t
        for t in times[times>=end-.000001]:
            if t-end>.08:break
            end=t+step
        na=max(lo,start-.075);nb=min(hi,end+.12)
        retained=[w for w in words if min(b,w['end'])-max(a,w['start'])>.015]
        if retained:
            # Quiet consonants and endings may be below the activity threshold.
            # Audio energy cannot overrule surviving aligned word spans.
            na=max(lo,min(na,min(w['start'] for w in retained)-.075))
            nb=min(hi,max(nb,max(w['end'] for w in retained)+.12))
        # Large disagreement needs review, not an aggressive automatic rewrite.
        if abs(na-a)>.6:na=a
        if abs(nb-b)>.6:nb=b
        # Preserve selected gameplay at either edge, even with a quiet microphone.
        if any(x<max(a,na)+.01 and y>min(a,na) for x,y in events):na=a
        if any(x<max(b,nb) and y>min(b,nb)-.01 for x,y in events):nb=b
        if nb-na<.1:na,nb=a,b
        result.append([na,nb])
        if abs(na-a)>.015 or abs(nb-b)>.015:
            changes.append(dict(before=[a,b],after=[na,nb],reason='Microphone energy edge refinement'))
    return result,changes
