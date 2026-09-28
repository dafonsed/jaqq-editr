"""Explicit safe suite: no model downloads, Resolve calls, or private recordings.

Do not blanket-collect the historical test_*.py files: some are live Resolve
integration scripts and some depend on private media or real local models.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SCRIPTS=[
    'test_speech_repairs_regression.py','test_pause_boundaries.py',
    'test_pipeline_audit.py','test_export_integrity_regressions.py',
    'test_dialogue_flow.py','test_attempt_review.py','test_repeated_openings.py',
    'test_script_review.py','test_automatic_cut.py','test_speech_edges.py',
    'test_dialogue_upgrade.py','test_edit_integrity.py','test_speech_recheck.py',
    'test_fresh_analysis.py','test_retention_only.py','test_ordered_export.py',
    'test_action_sequences.py','test_ui_states.py','test_launcher_setup.py',
]


def main():
    dependencies=[] if Path(sys.prefix).resolve()==ROOT/'.venv' else [str(ROOT/'.runtime-deps')]
    environment=dict(os.environ,PYTHONPATH=os.pathsep.join(dependencies+[str(ROOT)]),
                     QT_QPA_PLATFORM='offscreen',QT_LOGGING_RULES='qt.multimedia.*=false')
    rows=[]
    for filename in SCRIPTS:
        start=time.monotonic()
        try:
            result=subprocess.run([sys.executable,str(ROOT/filename)],cwd=ROOT,env=environment,
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            row=dict(file=filename,status='PASS' if result.returncode==0 else 'FAIL',
                returncode=result.returncode,output=result.stdout+result.stderr)
        except subprocess.TimeoutExpired:
            row=dict(file=filename,status='FAIL',output='Offline check exceeded 90 seconds')
        row['seconds']=round(time.monotonic()-start,3);rows.append(row)
        print(row['status'],filename,flush=True)
        if row['status']=='FAIL':print(row['output'],flush=True)
    folder=ROOT/'analysis'/'verification';folder.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Offline synthetic and policy regression checks; no live models or Resolve render',
        checks=rows,passed=sum(row['status']=='PASS' for row in rows),total=len(rows),
        real_source_playback='NOT VERIFIED',real_speech_listening='NOT VERIFIED',resolve_render='NOT VERIFIED')
    (folder/'offline-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['passed']!=report['total'])


if __name__=='__main__':raise SystemExit(main())
