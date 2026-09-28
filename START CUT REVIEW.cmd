@echo off
setlocal
cd /d "%~dp0"
rem Run the audited source. A previously built executable does not contain source fixes.
if defined RETENTION_PYTHON goto launch
if exist "%~dp0.venv\Scripts\pythonw.exe" set "RETENTION_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if defined RETENTION_PYTHON goto launch
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe" set "RETENTION_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
if defined RETENTION_PYTHON goto launch
for /f "delims=" %%P in ('where pythonw.exe 2^>nul') do if not defined RETENTION_PYTHON set "RETENTION_PYTHON=%%P"
if defined RETENTION_PYTHON goto launch
echo Python was not found. Set RETENTION_PYTHON to your Python executable.
echo See AUDIT_AND_FIX_REPORT.md for runtime and model requirements.
pause
exit /b 1
:launch
start "" "%RETENTION_PYTHON%" "%~dp0launch_cut_review.py"
