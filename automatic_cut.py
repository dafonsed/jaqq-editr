"""Local automatic rough-cut selection and export. No target runtime.

Tight speech, conservative retakes and classified gunfire with visual screen checks.
"""
import hashlib,json,time,subprocess,wave,uuid,math
from pathlib import Path
from app_paths import ROOT,add_dependencies
add_dependencies()
import av
import numpy as np
from review_core import read_transcript,save_json,merge_intervals,export_review
from analyze_speech import extract

class Cancelled(Exception):pass

def probe(source):
    source=Path(source)
    with av.open(str(source)) as c:
        if not c.streams.video or not c.streams.audio:
            raise ValueError('Choose a video recording with audio.')
        audio=[s.metadata.get('name',f'Audio track {i+1}') for i,s in enumerate(c.streams.audio)]
        video=c.streams.video[0]
        duration=float(video.duration*video.time_base) if video.duration is not None else c.duration/av.time_base
        fps=float(video.average_rate or video.base_rate or 0)
        if not math.isfinite(duration) or duration<=0 or not math.isfinite(fps) or fps<=0:
            raise ValueError('The recording has no reliable duration or frame rate.')
    return duration,fps,audio

def key_for(source):
    p=Path(source);s=p.stat()
    return hashlib.sha1(f'{p.resolve()}:{s.st_size}:{s.st_mtime_ns}'.encode()).hexdigest()[:14]

def join_near(ranges,duration,gap=.55):
    result=[]
    for a,b in merge_intervals(ranges,0,duration):
        if result and a-result[-1][1]<=gap:result[-1][1]=b
        else:result.append([a,b])
    return result

def select_ranges(transcript,events,duration,protected=()):
    from dialogue_cut import clean_dialogue,subtract
    speech,exclusions,duplicates=clean_dialogue(transcript,duration)
    # Alignment uncertainty and explicit visual/intentional-pause annotations
    # override text-only deletions. Do not run these spans through gunfire gating.
    duplicates=[r for r in duplicates if not any(r['start']<y and r['end']>x for x,y in protected)]
    exclusions=[(r['start'],r['end']) for r in duplicates]
    # Keep brief within-action gaps connected rather than creating flash shots.
    from action_sequences import action_sequences
    events=action_sequences(events,speech,duration)
    kept=join_near(speech+events+list(protected),duration,.10)
    words=[w for seg in transcript for w in seg['words'] if 0<w['end']-w['start']<3]
    for r in kept:
        for w in words:
            if w['start']<r[0]<w['end']:r[0]=max(0,w['start']-.09)
            if w['start']<r[1]<w['end']:r[1]=min(duration,w['end']+.12)
    # Subtract last: combat cannot put an abandoned take back into the edit.
    selected=subtract(join_near(kept,duration,.10),exclusions)
    # Removing a take can leave only its trailing padding as a tiny picture clip.
    # Keep a residual only when it contains actual speech or selected gameplay.
    selected=[(a,b) for a,b in selected if
        any(min(b,w['end'])-max(a,w['start'])>.025 for w in words) or
        any(min(b,y)-max(a,x)>.025 for x,y in list(events)+list(protected))]
    return selected,speech,duplicates

def plan(source,mic,progress=lambda s:None,cancel=lambda:False,*,intensity='balanced',protected_pauses=()):
    source=Path(source);duration,fps,names=probe(source)
    if not isinstance(mic,int) or not 0<=mic<len(names):raise ValueError('Choose a valid microphone track.')
    from pause_cleanup import pause_settings
    settings=pause_settings(intensity)
    protected_pauses=list(protected_pauses)
    for pause in protected_pauses:
        if not isinstance(pause,dict) or not pause.get('reason'):
            raise ValueError('Each protected pause needs source start/end and an editorial reason.')
        if not 0<=pause['start']<pause['end']<=duration:raise ValueError('Protected pause lies outside the recording.')
    (ROOT/'analysis').mkdir(parents=True,exist_ok=True)
    # Every button press gets a new analysis namespace. No transcript, extracted
    # analysis audio, gunfire scores or screen checks can come from an older run.
    run_id=uuid.uuid4().hex
    source_identity=key_for(source)
    key=f'{source_identity}-run-{run_id}'
    progress('Starting fresh analysis — re-listening and rechecking this recording…')
    def check():
        if cancel():raise Cancelled()
    cache=ROOT/'analysis'/f'transcript-{key}-{mic}.json'
    transcript=None
    if transcript is None:
        progress('Listening to your microphone…');wav=cache.with_suffix('.wav')
        extract(source,mic,0,duration,wav);check()
        model_dir=next((p.parent for p in (ROOT/'.model-cache').rglob('model.bin')
            if (p.parent/'config.json').is_file() and (p.parent/'tokenizer.json').is_file()),None)
        preferred=ROOT/'.model-cache/speech-small.en'
        if all((preferred/name).is_file() for name in ('model.bin','config.json','tokenizer.json')):model_dir=preferred
        if model_dir is None:raise ValueError('The speech model is missing from the app folder.')
        from faster_whisper import WhisperModel
        model=WhisperModel(str(model_dir),device='cpu',compute_type='int8',cpu_threads=4)
        segments,_=model.transcribe(str(wav),language='en',vad_filter=True,word_timestamps=True,
                                    beam_size=5,condition_on_previous_text=True,
                                    initial_prompt='Verbatim speech, including hesitations, repeated words, false starts and self-corrections.')
        transcript=[]
        for s in segments:
            check();progress(f'Listening to commentary: {int(s.end)//60}:{int(s.end)%60:02} / {int(duration)//60}:{int(duration)%60:02}')
            transcript.append(dict(start=s.start,end=s.end,text=s.text,words=[dict(start=w.start,end=w.end,text=w.word,probability=w.probability) for w in s.words]))
        save_json(cache,transcript)
    from speech_recheck import recover
    transcript,transcript_rechecks=recover(transcript,cache.with_suffix('.wav'),model,progress,cancel)
    save_json(ROOT/'analysis'/f'word-rechecks-{key}.json',transcript_rechecks)
    from speech_evidence import audit as audit_evidence,output_locations
    uncertain_audio,evidence_flags=audit_evidence(transcript,cache.with_suffix('.wav'),duration)
    save_json(ROOT/'analysis'/f'recovered-transcript-{key}.json',transcript)
    from transcript_quality import clean as clean_transcript
    transcript,transcript_warnings=clean_transcript(transcript)
    check();progress('Checking shooting and filtering menus/loading…')
    game=next((i for i,n in enumerate(names) if i!=mic and ('system' in n.lower() or 'game' in n.lower())),next((i for i in range(len(names)) if i!=mic),None))
    events=[];checks={'method':'No separate game track; commentary only'}
    if game is not None:
        from combat_detection import detect
        wav=ROOT/'analysis'/f'action-audio-{key}-{game}.wav'
        if not wav.is_file():extract(source,game,0,duration,wav)
        check()
        events,checks=detect(source,wav,f'{key}-{game}',duration,progress,cancel)
    check();progress('Shaping phrase timing, removing stutters and comparing alternate takes…')
    protected_context=uncertain_audio+[(p['start'],p['end']) for p in protected_pauses]
    kept,speech,duplicates=select_ranges(transcript,events,duration,protected=protected_context)
    from action_sequences import action_sequences
    selected_events=action_sequences(events,speech,duration)
    protected_context=selected_events+protected_context
    if not kept:raise ValueError('No clear commentary or shooting was found. Check the voice-track selection.')
    check();progress('Checking cut edges against your microphone waveform…')
    from speech_edges import refine_edges
    kept,edge_changes=refine_edges(kept,cache.with_suffix('.wav'),protected_context,
                                  [(r['start'],r['end']) for r in duplicates],transcript=transcript)
    check();progress('Removing sustained dead air inside the selected clips…')
    from pause_cleanup import clean_pauses,bridge_word_gaps
    kept,pause_bridges=bridge_word_gaps(kept,transcript,
        exclusions=[(r['start'],r['end']) for r in duplicates],protected_pauses=protected_context)
    kept,pause_changes=clean_pauses(kept,cache.with_suffix('.wav'),protected_context,
        transcript=transcript,settings=settings,exclusions=[(r['start'],r['end']) for r in duplicates],
        protected_pauses=protected_pauses)
    if not kept:raise ValueError('No audible commentary remained. Check the microphone track.')
    kept=[[round(a*fps)/fps,min(duration,round(b*fps)/fps)] for a,b in kept if round(b*fps)>round(a*fps)]
    from cut_integrity import protect_words
    kept,pre_review_edges=protect_words(kept,transcript,[(r['start'],r['end']) for r in duplicates],duration,fps,
        pre_roll=settings['leading'],post_roll=settings['trailing'])
    check();progress('Reviewing the selected dialogue in playback order…')
    from script_review import review_script
    check();progress('Comparing nearby talking points with the local semantic model…')
    kept,script_review=review_script(transcript,kept,protected_context,semantic=True,progress=progress,cancel=cancel)
    check();progress('Validating retained words and final frame boundaries…')
    exclusions=[(r['start'],r['end']) for r in duplicates]
    exclusions.extend((r['source_start'],r['source_end']) for r in script_review['changes'])
    kept,integrity_changes=protect_words(kept,transcript,exclusions,duration,fps,
        pre_roll=settings['leading'],post_roll=settings['trailing'])
    # Re-render the report against the actual final frames without rerunning any
    # deletion stage. Editorial changes and model evidence remain attached.
    _,final_report=review_script(transcript,kept,protected_context,audit_only=True)
    script_review['script']=final_report['script']
    # The final geometry audit must not erase semantic uncertainty or model warnings.
    script_review['flags']=output_locations(script_review['flags']+final_report['flags'],kept)
    from pause_cleanup import audit_pauses
    pause_flags=audit_pauses(kept,cache.with_suffix('.wav'),selected_events,transcript=transcript,
        settings=settings,exclusions=exclusions,protected_pauses=protected_pauses)
    review_flags=output_locations(evidence_flags+transcript_warnings+pause_flags,kept)
    from edit_manifest import build_manifest,cut_signature
    signature=cut_signature(kept,fps)
    manifest=build_manifest(kept,duration,fps,duplicates+script_review['changes']+pause_changes,
        review_flags+script_review['flags'])
    if key_for(source)!=source_identity:
        raise ValueError('The recording changed during analysis. Finish recording, then try again.')
    return dict(source=str(source),duration=duration,fps=fps,kept=kept,
                speech=speech,action=selected_events,raw_action=events,microphone=mic,game_audio=game,
                speech_model=str(model_dir),source_key=source_identity,settings=settings,protected_pauses=protected_pauses,
                evidence_flags=review_flags,edit_manifest=manifest,
                transcript_warnings=transcript_warnings,
                transcript_rechecks=transcript_rechecks,
                duplicate_takes=len(duplicates),removed_duplicates=duplicates,combat_checks=checks,
                target_duration=None,analysis_run_id=run_id,analysis_mode='fresh',
                pipeline_version='speech-integrity-2',cut_signature=signature,integrity_adjustments=pre_review_edges+integrity_changes,
                script_review=script_review,edge_adjustments=edge_changes,pause_removals=pause_changes,pause_context_spans=pause_bridges,
                selection_method='Evidence-gated speech repair, configurable contextual pause budgets, nearby alternate-take review, and sampled gameplay selection. Uncertain audio is retained and flagged. This does not evaluate the full story or verify rendered joins.')

FUSCRIPT=Path(r'C:\Program Files\Blackmagic Design\DaVinci Resolve\fuscript.exe')

def check_resolve():
    p=ROOT/'analysis'/'auto-resolve-preflight.lua'
    p.parent.mkdir(parents=True,exist_ok=True)
    if not FUSCRIPT.is_file():raise ValueError('DaVinci Resolve scripting is unavailable on this computer.')
    p.write_text('local r=bmd.scriptapp("Resolve"); assert(r,"Open Resolve first"); local p=r:GetProjectManager():GetCurrentProject(); assert(p,"Open a project"); print("AUTO_READY")',encoding='utf-8')
    try:
        r=subprocess.run([str(FUSCRIPT),'-l','lua',str(p)],capture_output=True,text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise ValueError('Resolve is not responding yet. Bring Resolve to the front, close any dialogs, then click Create automatic cut again.') from None
    if 'AUTO_READY' not in r.stdout:raise ValueError('Open a project in DaVinci Resolve, then click Create automatic cut again.')

def send(result,progress=lambda s:None):
    """Export only selected picture and original linked audio."""
    from review_media import audio_sidecars
    from cut_integrity import validate
    validate(result['kept'],result['duration'],result['fps'])
    from edit_manifest import cut_signature
    if result.get('cut_signature')!=cut_signature(result['kept'],result['fps']):
        raise ValueError('The cut plan changed after review; analyze it again before export.')
    if result.get('source_key')!=key_for(result['source']):
        raise ValueError('The source recording changed after analysis; analyze it again before export.')
    try:previous=json.loads((ROOT/'analysis'/'last-automatic-cut.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):previous=None
    if isinstance(previous,dict) and previous.get('source')==result['source']:
        before=[[round(a*result['fps']),round(b*result['fps'])] for a,b in previous['kept']]
        after=[[round(a*result['fps']),round(b*result['fps'])] for a,b in result['kept']]
        result=dict(result,comparison=dict(previous_run=previous.get('analysis_run_id'),
            identical_frame_ranges=before==after,previous_clips=len(before),current_clips=len(after),
            added_ranges=len(set(map(tuple,after))-set(map(tuple,before))),
            removed_ranges=len(set(map(tuple,before))-set(map(tuple,after)))))
    folder=ROOT/'exports'/('Retention cut '+time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%100000))
    audio=audio_sidecars(result['source'],progress)
    result={k:v for k,v in result.items() if not k.startswith(('sfx_','caption','motion_'))}
    export_review(folder,result['source'],result['duration'],result['fps'],[],{},audio=audio,
        keep_ranges=result['kept'],timeline_prefix='RETENTION - '+Path(result['source']).stem[:35]+' - ',
        draft_note='Retention cut: original picture and linked voice/game audio. Review dialogue joins and action context.')
    save_json(folder/'automatic-plan.json',result)
    save_json(folder/'edit-manifest.json',result.get('edit_manifest',{}))
    if result.get('script_review'):
        from script_review import readable
        save_json(folder/'script-review.json',result['script_review'])
        (folder/'selected-script.txt').write_text(readable(result['script_review']),encoding='utf-8')
    progress('Opening your retention cut in Resolve…')
    script=folder/'create_resolve_draft.lua'
    backup=(folder/'Automatic cut.drt').as_posix()
    with script.open('a',encoding='utf-8') as f:
        f.write('\nassert(timeline:GetTrackCount("subtitle")==0,"Unexpected subtitle track")\n')
        f.write('for _,item in ipairs(videoItems) do assert(item:GetFusionCompCount()==0,"Unexpected Fusion effect") end\n')
        f.write('timeline:SetCurrentTimecode(timeline:GetStartTimecode())\n')
        f.write('assert(timeline:Export([==['+backup+']==],resolve.EXPORT_DRT),"Backup failed")\n')
        f.write('assert(project:SetCurrentTimeline(timeline),"Could not activate completed cut")\n')
        f.write('assert(resolve:GetProjectManager():SaveProject(),"Cut created, but project save failed")\n')
        f.write('assert(project:GetCurrentTimeline():GetName()==timeline:GetName(),"Completed cut is not active")\n')
        f.write('print("AUTO_VERIFIED "..#videoItems.." "..expectedDuration.." "..#audioClips)\n')
    r=subprocess.run([str(FUSCRIPT),'-l','lua',str(script)],capture_output=True,text=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    output=r.stdout+'\n'+r.stderr
    (folder/'resolve-output.txt').write_text(output,encoding='utf-8')
    if 'AUTO_VERIFIED ' not in output:
        raise ValueError('Resolve did not confirm the import. Check for a RETENTION timeline before retrying. Details: '+str(folder/'resolve-output.txt'))
    name=output.split('CUT_REVIEW_OK ',1)[1].splitlines()[0]
    result=dict(result,folder=str(folder),timeline=name,clips=len(result['kept']),
        edited_duration=sum(round(b*result['fps'])-round(a*result['fps']) for a,b in result['kept'])/result['fps'],
        timeline_verified=True,render_verified=False,verified=False)
    review_file=folder/'review.json'
    if review_file.is_file():
        exported=json.loads(review_file.read_text(encoding='utf-8'))
        exported['verification']['timeline_verified']=True
        save_json(review_file,exported)
    save_json(folder/'automatic-plan.json',result)
    save_json(ROOT/'analysis'/'last-automatic-cut.json',result)
    return result
