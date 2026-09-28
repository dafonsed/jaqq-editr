"""Inspect already-rendered synthetic evaluation files without API/Resolve calls."""
import argparse
import json
import math
from pathlib import Path
import time

from app_paths import ROOT,add_dependencies
add_dependencies()
import av
import numpy as np
from review_core import build_frame_map,map_source_span,save_json
from review_render import audit_render,transcribe_render


def _read_picture(path,position):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        origin=float((stream.start_time or 0)*stream.time_base)
        container.seek(max(0,int((origin+position)*av.time_base)))
        for frame in container.decode(stream):
            if float(frame.pts*frame.time_base)-origin>=position-.001:
                return frame
    raise ValueError('No frame decoded at the requested inspection position.')


def inspect_pictures(source,output,frame_map,fps,folder):
    comparisons=[]
    for row in frame_map:
        offset=min(2,row['duration_frames']-1)
        source_time=(row['source_start_frame']+offset)/fps
        output_time=(row['output_start_frame']+offset)/fps
        before=_read_picture(source,source_time)
        after=_read_picture(output,output_time)
        original=before.reformat(width=after.width,height=after.height,format='rgb24').to_ndarray()
        rendered=after.to_ndarray(format='rgb24')
        difference=float(np.mean(np.abs(original.astype(float)-rendered.astype(float))))
        comparisons.append(dict(clip=row['clip_index'],source_time=source_time,output_time=output_time,
            mean_channel_error=difference,status='MATCH' if difference<7 else 'REVIEW'))
        if row['clip_index']==1:
            before.to_image().save(folder/'source-frame.png')
            after.to_image().save(folder/'rendered-frame.png')
    return comparisons


def plan_checks(plan,output,frame_map):
    from automatic_cut import key_for
    from edit_manifest import cut_signature
    exported=json.loads(Path(output).with_suffix(Path(output).suffix+'.render.json').read_text(encoding='utf-8'))
    checks=dict(source_identity_matches=plan['source_key']==key_for(plan['source']),
        approved_cut_signature_matches=plan['cut_signature']==cut_signature(plan['kept'],plan['fps']),
        render_uses_approved_frame_map=exported['frame_map']==frame_map,
        render_uses_approved_source=Path(exported['source']).resolve()==Path(plan['source']).resolve())
    return dict(status='PASS' if all(checks.values()) else 'FAIL',**checks)


def contact_sheet(folder):
    """Overview of decoded source/output frames for actual visual inspection."""
    from PIL import Image,ImageDraw,ImageFont
    report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15)
    rows=math.ceil(len(report['cases'])/3)
    sheet=Image.new('RGB',(1800,rows*240+48),'#111821')
    draw=ImageDraw.Draw(sheet)
    draw.text((18,10),'SYNTHETIC MEDIA · decoded source (left) / rendered edit (right) · listening not verified',font=font,fill='white')
    for i,result in enumerate(report['cases']):
        x,y=(i%3)*600+12,(i//3)*240+48
        title=result['case'].replace('-ai-plan','')
        draw.text((x,y),title+' · timing '+result['qa']['geometry']+' · words '+result['qa']['word_check']['status'],font=font,fill='#bcd8ef')
        for column,name in enumerate(('source-frame.png','rendered-frame.png')):
            picture=Image.open(folder/result['case']/name).convert('RGB')
            picture.thumbnail((282,170))
            sheet.paste(picture,(x+column*294,y+30))
        draw.text((x,y+207),f"{result['qa']['video_frames']} output frames · {len(result['qa']['issues'])} measured flags",font=small,fill='#afbbc7')
    target=folder/'source-render-contact-sheet.png'
    sheet.save(target)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'analysis'/'ai-evaluation')
    parser.add_argument('--render-folder',type=Path,help='Use a completed immutable render run with the supplied plans')
    parser.add_argument('--speech-check',action='store_true')
    args=parser.parse_args()
    plans=sorted(args.folder.glob('*-ai-plan.json'))
    if not plans:raise ValueError('No synthetic evaluation plans are available.')
    model=None
    if args.speech_check:
        from faster_whisper import WhisperModel
        model=WhisperModel(str(ROOT/'.model-cache'/'speech-small.en'),device='cpu',compute_type='int8',cpu_threads=4)
    results=[]
    for plan_path in plans:
        started=time.monotonic()
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        output=(args.render_folder or args.folder)/plan_path.name.replace('-ai-plan.json','-ai.mp4')
        folder=(args.render_folder or args.folder)/'independent-review'/plan_path.stem
        folder.mkdir(parents=True,exist_ok=True)
        frame_map=build_frame_map(plan['kept'],plan['duration'],plan['fps'])
        expected=[]
        for segment in plan['transcript']:
            for word in segment['words']:
                expected.extend(dict(word,start=span['output_start'],end=span['output_end'])
                    for span in map_source_span(frame_map,word['start'],word['end']))
        transcription=transcribe_render(output,plan['microphone'],frame_map[-1]['output_end'],model=model,
            progress=lambda message:print(message,flush=True)) if args.speech_check else None
        qa=audit_render(output,frame_map,plan['fps'],microphone=plan['microphone'],expected_words=expected,
            observed_words=transcription['words'] if transcription else None)
        pictures=inspect_pictures(plan['source'],output,frame_map,plan['fps'],folder)
        result=dict(case=plan_path.stem,qa=qa,picture_correspondence=pictures,
            plan_source_checks=plan_checks(plan,output,frame_map),seconds=round(time.monotonic()-started,3),listening_verified=False)
        save_json(folder/'inspection.json',result);results.append(result)
        print(result['case'],qa['geometry'],qa['word_check']['status'],result['seconds'],flush=True)
    save_json((args.render_folder or args.folder)/'independent-review'/'summary.json',dict(scope='Synthetic media independent local ASR and decoded-picture inspection',
        api_calls=0,human_preference='NOT MEASURED',listening='NOT VERIFIED',cases=results))
    print(contact_sheet((args.render_folder or args.folder)/'independent-review'),flush=True)


if __name__=='__main__':main()
