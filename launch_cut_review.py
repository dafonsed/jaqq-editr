"""Visible error handling for launches from Windows Explorer."""
import ctypes
import faulthandler
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from launch_status import STATUS_ENV, report_startup

# pythonw normally sets these streams to None. Preserve Python tracebacks and
# fatal native errors for both direct launches and the supervised bootstrap.
folder=ROOT/'analysis';folder.mkdir(exist_ok=True)
_diagnostic_stream=(folder/'application.log').open('a',encoding='utf-8',buffering=1)
if sys.stdout is None:sys.stdout=_diagnostic_stream
if sys.stderr is None:sys.stderr=_diagnostic_stream
faulthandler.enable(file=_diagnostic_stream,all_threads=True)
try:
    runpy.run_path(str(ROOT/'automatic_review.py'),run_name='__main__')
except SystemExit as e:
    if e.code not in (None,0):
        report_startup('error',error=f'The editor exited with code {e.code}')
        print(f'The editor exited with code {e.code}',file=sys.stderr,flush=True)
        if '--launch-check' not in sys.argv and STATUS_ENV not in os.environ:
            ctypes.windll.user32.MessageBoxW(None,'Cut Review could not start. Details are in analysis/application.log.','Cut Review',0x10)
    raise
except BaseException:
    details=traceback.format_exc()
    folder=ROOT/'analysis';folder.mkdir(exist_ok=True)
    (folder/'launch-error.log').write_text(details,encoding='utf-8')
    report_startup('error',error=details.splitlines()[-1])
    print(details,file=sys.stderr,flush=True)
    message=('Cut Review could not open.\n\n'+details.splitlines()[-1]+
        '\n\nRun START CUT REVIEW.cmd to install/check the local runtime.'+
        '\nDetails: '+str(folder/'launch-error.log'))
    if '--launch-check' in sys.argv or STATUS_ENV in os.environ:
        if sys.stderr:print(message,file=sys.stderr)
    else:
        ctypes.windll.user32.MessageBoxW(None,message,'Cut Review — could not start',0x10)
    sys.exit(1)
