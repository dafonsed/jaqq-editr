"""Review data and Resolve export for reviewed cuts or explicit ordered draft plans."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

from app_paths import ROOT

def merge_intervals(intervals, start=0, end=float('inf')):
    result = []
    for a, b in sorted((max(start, a), min(end, b)) for a, b in intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1] + 1e-6:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result

def complement(intervals, start, end):
    result, cursor = [], start
    for a, b in merge_intervals(intervals, start, end):
        if a > cursor:
            result.append([cursor, a])
        cursor = b
    if cursor < end:
        result.append([cursor, end])
    return result

def stamp(t):
    t = max(0, t)
    return f'{int(t)//60:02}:{t%60:05.2f}'

def row(kind, start, end, text, **extra):
    key = f'{kind}:{start:.3f}:{end:.3f}'
    return dict(id=hashlib.sha1(key.encode()).hexdigest()[:14], kind=kind,
                start=round(start, 3), end=round(end, 3), text=text, **extra)

def build_rows(transcript, duration):
    words = sorted([w for s in transcript for w in s.get('words', [])
                    if 0 <= w['start'] < w['end'] <= duration+0.1], key=lambda w:w['start'])
    rows, group = [], []
    def finish():
        if group:
            rows.append(row('Speech', group[0]['start'], group[-1]['end'],
                            ''.join(w['text'] for w in group).strip()))
    for w in words:
        if group and (w['start']-group[-1]['end'] > 1.1 or
                      (len(group) >= 12 and group[-1]['text'].strip().endswith(('.', '?', '!')))):
            finish(); group=[]
        group.append(w)
    finish()
    # Word-level times are essential: one transcript segment may span a long VAD gap.
    speech = merge_intervals([(w['start']-.4, w['end']+.4) for w in words], 0, duration)
    for a, b in complement(speech, 0, duration):
        if b-a < 3:
            continue
        # Small review cards; no implication that silence is expendable.
        count = max(1, math.ceil((b-a)/15))
        for i in range(count):
            lo, hi = a+(b-a)*i/count, a+(b-a)*(i+1)/count
            rows.append(row('Quiet mic', lo, hi,
                'No transcribed words here. Check gameplay and game audio before cutting.'))
    tokens = [re.sub(r'[^a-z0-9]', '', w['text'].lower()) for w in words]
    seen, repeated = {}, []
    for i in range(len(tokens)-4):
        phrase = tuple(tokens[i:i+5])
        if not all(phrase):
            continue
        j = seen.get(phrase)
        if j is not None and i-j >= 5 and 0 < words[i]['start']-words[j]['start'] < 35:
            a, b = words[j]['start'], words[i+4]['end']
            if not any(abs(a-r['start']) < 2 for r in repeated):
                repeated.append(row('Similar takes', a, b,
                    'Repeated wording: “'+' '.join(phrase)+'”. Compare both takes; repetition may be intentional.',
                    suggested_end=max(a+.01,words[i]['start']-.15)))
        seen[phrase] = i
    rows.extend(repeated)
    return sorted(rows, key=lambda r:(r['start'], r['kind']))

def reference_rows():
    path=ROOT/'analysis'/'main-timeline.json'
    rows=[]
    if not path.exists():
        return rows
    # Restrict the calibration to the known eight consecutive base-video selections.
    try: clips=json.loads(path.read_text())
    except (ValueError,OSError):return []
    for c in clips:
        if c['type'] != 'Sm2TiVideoClip' or not c.get('In'):
            continue
        t=(int(c['Start'])-216000)/60
        if 57.38 <= t < 79.05 and (c.get('MediaFilePath') or '').endswith('22.30.24.03.mp4'):
            a=int(c['In'])/60; b=a+int(c['Duration'])/60
            rows.append(row('Your edit',a,b, f'Kept in your finished video at {stamp(t)}.', edit_start=t))
    return rows

def read_transcript(path):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(data,list) and all(isinstance(r,dict) and isinstance(r.get('words'),list) for r in data):return data
    except (ValueError,OSError):pass
    return None

def save_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(path)


def build_frame_map(kept, duration, fps):
    """The single source of truth for half-open source/output frame ranges.

    Keep Python's rounding convention used by the selection engine. Resolve must
    consume these integers, rather than independently rounding source seconds.
    Editorial order and intentional repeated selections are preserved.
    """
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) and x > 0 for x in (duration, fps)):
        raise ValueError('Duration and frame rate must be finite positive numbers.')
    result, cursor = [], 0
    for index, item in enumerate(kept, 1):
        if (len(item) != 2 or not all(isinstance(x, (int, float)) and
                not isinstance(x, bool) and math.isfinite(x) for x in item)
                or not 0 <= item[0] < item[1] <= duration):
            raise ValueError('Every draft range must lie within the recording.')
        a, b = item
        first, last = round(a * fps), round(b * fps)
        if last <= first:
            raise ValueError('Every draft range must contain at least one frame.')
        count = last - first
        result.append(dict(clip_index=index, requested_source_start=a,
            requested_source_end=b, source_start_frame=first,
            source_end_frame=last, source_start=first/fps, source_end=last/fps,
            output_start_frame=cursor, output_end_frame=cursor+count,
            output_start=cursor/fps, output_end=(cursor+count)/fps,
            duration_frames=count))
        cursor += count
    if not result:
        raise ValueError('No footage remains. Restore at least one section.')
    return result


def map_source_span(frame_map, start, end):
    """Project a source span onto every retained occurrence, including repeats."""
    mapped = []
    for clip in frame_map:
        a, b = max(start, clip['source_start']), min(end, clip['source_end'])
        if b > a:
            mapped.append(dict(clip_index=clip['clip_index'], source_start=a,
                source_end=b, output_start=clip['output_start']+a-clip['source_start'],
                output_end=clip['output_start']+b-clip['source_start'],
                partial=a > start or b < end))
    return mapped


def export_review(folder, source, duration, fps, rows, decisions, cuts=None, audio=None,
                  *, keep_ranges=None, timeline_prefix='Cut Review draft ', draft_note=None):
    if keep_ranges is not None:
        if cuts is not None:
            raise ValueError('Supply either removals or an ordered keep plan, not both.')
        keep_ranges = [list(r) for r in keep_ranges]
    # Validate before creating artifacts, including the manual-cut export path.
    build_frame_map([[0, duration]], duration, fps)
    selected=[dict(r,decision=decisions.get(r['id'],'Unreviewed')) for r in rows]
    removals=cuts if cuts is not None else [(r['start'],r['end']) for r in selected if r['decision']=='Cut']
    if not all(len(r)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in r)
               and r[1]>r[0] for r in removals):
        raise ValueError('Removal ranges must contain finite increasing times.')
    removed=merge_intervals(removals,0,duration)
    kept=complement(removed,0,duration)
    if keep_ranges is not None:
        kept=keep_ranges  # Preserve editorial order, including an optional cold open.
        removed=complement(kept,0,duration)
    frame_map=build_frame_map(kept,duration,fps)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'review.json',dict(source=str(source),duration=duration,fps=fps,
                rows=selected,removed=removed,kept=kept,frame_map=frame_map,
                frame_convention='Half-open [start, end), output starts at zero',
                edited_duration=frame_map[-1]['output_end'],audio=audio or [],
                verification=dict(plan_validated=True,timeline_verified=False,
                    rendered_media_verified=False,audio_joins_listened=False),
                audio_smoothing='Not applied; review hard cuts for audible artifacts.'))
    with (folder/'decisions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['kind','start','end','decision','text'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in selected)
    # Lua runs inside Resolve's built-in console, independent of the host Python version.
    quote=lambda s:'[==['+str(s).replace(']==]','] = = ]')+']==]'
    ranges=',\n'.join('{%.6f, %.6f}'%(a,b) for a,b in kept)
    frame_ranges=',\n'.join('{%d, %d, %d, %d}' % (r['source_start_frame'],
        r['source_end_frame'],r['output_start_frame'],r['output_end_frame']) for r in frame_map)
    audio_lua=',\n'.join('{path='+quote(Path(x['path']).as_posix())+', name='+quote(x['name'])+'}' for x in (audio or []))
    script='''-- Generated by Cut Review. Creates a NEW draft timeline; never edits an existing one.
local resolve = bmd.scriptapp("Resolve")
assert(resolve, "Open DaVinci Resolve first.")
local project = resolve:GetProjectManager():GetCurrentProject()
assert(project, "Open a project first.")
local pool = project:GetMediaPool()
local source = SOURCE_PATH
local function normal(path) return string.lower(string.gsub(path or "", "\\\\", "/")) end
local function find(folder)
  for _, item in ipairs(folder:GetClipList()) do
    if normal(item:GetClipProperty("File Path")) == normal(source) then return item end
  end
  for _, child in ipairs(folder:GetSubFolderList()) do
    local found = find(child); if found then return found end
  end
end
local clip = find(pool:GetRootFolder())
if not clip then
  local imported = pool:ImportMedia({source})
  assert(imported and #imported > 0, "Could not import the source recording.")
  clip = imported[1]
end
local fps = tonumber(clip:GetClipProperty("FPS"))
assert(fps and fps > 0, "Resolve did not report the source frame rate.")
assert(math.abs(fps - EXPECTED_FPS) < 0.002, "Source frame rate differs from the review. Check it before importing cuts.")
local ranges = {KEEP_RANGES}
-- Source start/end and output start/end, all half-open. Rounded once in Python.
local frameRanges = {FRAME_RANGES}
local audioFiles = {AUDIO_FILES}
local audioClips = {}
for _, a in ipairs(audioFiles) do
  local imported = pool:ImportMedia({a.path})
  assert(imported and #imported > 0, "Could not import a separate audio stream.")
  table.insert(audioClips, imported[1])
end
assert(#ranges > 0, "No footage remains. Restore at least one section in Cut Review.")
local timeline = pool:CreateEmptyTimeline(TIMELINE_PREFIX .. os.date("%Y-%m-%d %H-%M-%S"))
assert(timeline, "Could not create a new timeline.")
project:SetCurrentTimeline(timeline)
assert(timeline:SetSetting("useCustomSettings", "1"), "Could not enable custom settings on new draft.")
assert(timeline:SetSetting("timelineFrameRate", tostring(EXPECTED_FPS)), "Could not set draft frame rate.")
local infos = {}
local origin = timeline:GetStartFrame()
local expectedDuration = frameRanges[#frameRanges][4]
for _, r in ipairs(frameRanges) do
  table.insert(infos, {mediaPoolItem=clip, startFrame=r[1], endFrame=r[2], mediaType=1, trackIndex=1, recordFrame=origin+r[3]})
end
local added = pool:AppendToTimeline(infos)
assert(added and #added == #infos, "Resolve did not add every video range to the new draft.")
local function checkTrack(kind, index)
  local items=timeline:GetItemListInTrack(kind,index) or {}
  table.sort(items,function(a,b) return a:GetStart()<b:GetStart() end)
  assert(#items==#frameRanges,"Track clip count differs from the approved plan.")
  for j,item in ipairs(items) do
    local r=frameRanges[j]
    assert(item:GetStart()==origin+r[3],"Timeline gap or overlap at clip "..j)
    assert(item:GetDuration()==r[4]-r[3],"Clip duration differs from the approved plan at clip "..j)
    assert(item:GetLeftOffset()==r[1],"Source in-point differs from the approved plan at clip "..j)
  end
  return items
end
local videoItems=checkTrack("video",1)
local audioTracks={}
for i, audioClip in ipairs(audioClips) do
  while timeline:GetTrackCount("audio") < i do assert(timeline:AddTrack("audio","stereo"), "Could not add audio track.") end
  timeline:SetTrackName("audio", i, audioFiles[i].name)
  local audioInfos = {}
  for _, r in ipairs(frameRanges) do
    table.insert(audioInfos,{mediaPoolItem=audioClip,startFrame=r[1],endFrame=r[2],mediaType=2,trackIndex=i,recordFrame=origin+r[3]})
  end
  local audioAdded=pool:AppendToTimeline(audioInfos)
  assert(audioAdded and #audioAdded==#audioInfos,"Resolve did not add every audio range.")
  local audioItems=checkTrack("audio",i)
  audioTracks[i]=audioItems
  assert(#audioItems==#videoItems,"Audio and video clip counts differ.")
  for j, v in ipairs(videoItems) do
    assert(v:GetStart()==audioItems[j]:GetStart() and v:GetDuration()==audioItems[j]:GetDuration(),"Audio and video timing differ; inspect the new draft.")
  end
end
for j, v in ipairs(videoItems) do
  local linked={v}
  for i=1,#audioClips do table.insert(linked,audioTracks[i][j]) end
  if #linked>1 then assert(timeline:SetClipsLinked(linked,true),"Could not link video and audio clips.") end
end
assert(timeline:GetEndFrame()-timeline:GetStartFrame()==expectedDuration,"Draft duration differs from the selected ranges.")
for i,r in ipairs(frameRanges) do
  print("CUT_REVIEW_RANGE "..i.." "..r[1].." "..r[2].." "..r[3].." "..r[4])
end
timeline:AddMarker(0, "Yellow", "Review draft", DRAFT_NOTE, 1)
resolve:OpenPage("edit")
print("Created " .. timeline:GetName() .. ". Check both source audio streams and cut boundaries before using it.")
print("CUT_REVIEW_OK " .. timeline:GetName())
'''.replace('SOURCE_PATH',quote(Path(source).as_posix())).replace('EXPECTED_FPS',str(fps)).replace('KEEP_RANGES',ranges).replace('FRAME_RANGES',frame_ranges).replace('AUDIO_FILES',audio_lua).replace('TIMELINE_PREFIX',quote(timeline_prefix)).replace('DRAFT_NOTE',quote(draft_note or 'Only manually marked cuts were removed. Independent audio streams are copied without re-encoding. Review audio joins; no smoothing, effects or sound design are recreated.'))
    (folder/'create_resolve_draft.lua').write_text(script,encoding='utf-8')
    return folder
