"""First-run source setup. Only a project-local venv is installed or modified."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PROBES={
    'PySide6': 'from PySide6.QtCore import QTimer; from PySide6.QtWidgets import QApplication; from PySide6.QtMultimedia import QMediaPlayer; from PySide6.QtNetwork import QLocalServer',
    'av': 'import av; assert callable(av.open)',
    'numpy': 'import numpy; assert numpy.zeros(1).size == 1',
    'Pillow': 'from PIL import Image; Image.new("RGB", (1,1))',
    'onnxruntime': 'from onnxruntime import InferenceSession',
    'tokenizers': 'from tokenizers import Tokenizer',
    'faster-whisper': 'from faster_whisper import WhisperModel; import ctranslate2',
}


def runtime_python(root):
    return Path(root).resolve()/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def clean_environment():
    environment=dict(os.environ)
    for name in ('PYTHONHOME','PYTHONPATH'):environment.pop(name,None)
    environment['PYTHONNOUSERSITE']='1'
    return environment


def audit_runtime(executable, *, runner=subprocess.run):
    executable=Path(executable)
    if not executable.is_file():
        return dict(ok=False,missing=['project Python runtime'],errors={})
    code='''import json
checks = PROBES
errors = {}
for name, statement in checks.items():
    try:
        exec(statement, {})
    except Exception as error:
        errors[name] = type(error).__name__ + ': ' + str(error)
print(json.dumps(dict(ok=not errors, missing=list(errors), errors=errors)))
'''.replace('PROBES',repr(PROBES))
    try:
        result=runner([str(executable),'-I','-c',code],capture_output=True,text=True,
            encoding='utf-8',errors='replace',env=clean_environment(),timeout=90)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        if type(report.get('ok')) is not bool or not isinstance(report.get('missing'),list):
            raise ValueError('Malformed dependency probe')
        if result.returncode and report['ok']:
            return dict(ok=False,missing=['runtime imports'],errors={'runtime':result.stderr[-4000:]})
        return report
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired) as error:
        return dict(ok=False,missing=['runtime imports'],errors={'runtime':str(error)})


def ensure_runtime(root, *, runner=subprocess.run, progress=print, force=False):
    root=Path(root).resolve()
    executable=runtime_python(root)
    manifest=root/'requirements-runtime.txt'
    log=root/'analysis'/'runtime-setup.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    def record(text):
        with log.open('a',encoding='utf-8') as handle:handle.write(text+'\n')
    def run_setup(command,label):
        progress(label)
        record(label)
        try:
            with log.open('a',encoding='utf-8') as handle:
                result=runner(command,cwd=str(root),env=clean_environment(),stdout=handle,
                    stderr=subprocess.STDOUT,timeout=900)
        except (OSError,subprocess.TimeoutExpired) as error:
            record(str(error))
            raise RuntimeError(f'{label} failed. Details: {log}') from error
        if result.returncode:
            raise RuntimeError(f'{label} failed (exit {result.returncode}). Details: {log}')
    if not manifest.is_file():raise RuntimeError(f'Missing runtime manifest: {manifest}')
    if not executable.is_file():
        if sys.version_info<(3,12) or struct.calcsize('P')!=8:
            raise RuntimeError('Install 64-bit Python 3.12 or newer, then run START CUT REVIEW.cmd again.')
        run_setup([sys.executable,'-m','venv',str(root/'.venv')],
                  'Creating the local Python environment...')
    report=audit_runtime(executable,runner=runner)
    record(json.dumps(report))
    if not report['ok'] or force:
        progress('First-run setup: downloading Python packages. This can take a few minutes.')
        progress(f'Installation details: {log}')
        command=[str(executable),'-m','pip','install','--disable-pip-version-check',
                 '--only-binary=:all:','--upgrade','-r',str(manifest)]
        # A broken native package can still have valid installation metadata;
        # pip must replace its files rather than saying "already satisfied".
        command.append('--force-reinstall')
        run_setup(command,'Installing the editor runtime...')
        report=audit_runtime(executable,runner=runner)
        record(json.dumps(report))
        if not report['ok']:
            raise RuntimeError('Runtime packages still cannot load: '+', '.join(report['missing'])+
                               f'. Details: {log}')
    return executable


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='Report dependency status without installing')
    mode.add_argument('--setup-only',action='store_true',help='Install dependencies without opening the app')
    mode.add_argument('--launch-check',action='store_true',help='Check the actual GUI startup and exit')
    parser.add_argument('--repair',action='store_true',help='Reinstall the local runtime packages')
    args,application_args=parser.parse_known_args(argv)
    if args.check:
        report=audit_runtime(runtime_python(ROOT))
        print(json.dumps(report,indent=2))
        return 0 if report['ok'] else 1
    try:
        executable=ensure_runtime(ROOT,force=args.repair,
            progress=lambda value:print(value,flush=True))
        if args.setup_only:
            print('Python runtime verified. No model files were downloaded.',flush=True)
            return 0
        launch=[str(executable),str(ROOT/'launch_cut_review.py')]
        if args.launch_check:
            result=subprocess.run(launch+['--launch-check']+application_args,cwd=ROOT,
                env=clean_environment(),timeout=60)
            if result.returncode:raise RuntimeError('The GUI launch check failed; see analysis/launch-error.log.')
            return 0
        windowed=executable.with_name('pythonw.exe')
        if os.name=='nt' and windowed.is_file():launch[0]=str(windowed)
        subprocess.Popen(launch+application_args,cwd=ROOT,env=clean_environment(),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return 0
    except (RuntimeError,OSError,subprocess.TimeoutExpired) as error:
        print(f'Could not start the editor: {error}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
