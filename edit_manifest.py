"""Inspectable source/output decisions using the exact frame plan sent to Resolve."""
import hashlib
import json
from review_core import complement,build_frame_map


def cut_signature(kept, fps):
    return hashlib.sha256(json.dumps([[round(a*fps),round(b*fps)] for a,b in kept]).encode()).hexdigest()


def build_manifest(kept, duration, fps, decisions, flags=()):
    clips = []
    frame_map=build_frame_map(kept,duration,fps)
    for row in frame_map:
        first,last=row['source_start_frame'],row['source_end_frame']
        cursor=row['output_start_frame']
        clips.append(dict(clip=row['clip_index'],source_start=first/fps,source_end=last/fps,
            source_in_frame=first,source_out_frame_exclusive=last,
            output_start=cursor/fps,output_end=(cursor+last-first)/fps,
            output_in_frame=cursor,output_out_frame_exclusive=cursor+last-first,
            reason='Retained speech, selected action or explicitly protected review context'))
    cursor=frame_map[-1]['output_end_frame']
    removed=[]
    for a,b in complement(kept,0,duration):
        evidence=[]
        for decision in decisions:
            x=decision.get('source_start',decision.get('start',-1))
            y=decision.get('source_end',decision.get('end',-1))
            if x<b and y>a:evidence.append(decision)
        removed.append(dict(source_start=a,source_end=b,
            output_join=sum(max(0,min(a,y)-x) for x,y in kept if x<a),
            reason='; '.join(dict.fromkeys(x['reason'] for x in evidence)) if evidence else
                'Outside retained speech/action and protected context; no semantic deletion inferred',
            confidence='See decision evidence' if evidence else 'Selection only; listening review required',
            evidence=evidence))
    return dict(version=1,fps=fps,cut_signature=cut_signature(kept,fps),
        output_frames=cursor,output_duration=cursor/fps,clips=clips,removed=removed,
        flags=list(flags),verification=dict(plan='PASS',render='NOT VERIFIED',listening='NOT VERIFIED'))
