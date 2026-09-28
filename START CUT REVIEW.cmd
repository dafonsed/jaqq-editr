@echo off
setlocal
cd /d "%~dp0"
rem The bootstrap installs and checks a local runtime before opening the GUI.
if defined RETENTION_PYTHON goto launch
if exist "%~dp0.venv\Scripts\python.exe" set "RETENTION_PYTHON=%~dp0.venv\Scripts\python.exe"
if defined RETENTION_PYTHON goto launch
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "RETENTION_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if defined RETENTION_PYTHON goto launch
for /f "delims=" %%P in ('where python.exe 2^>nul') do if not defined RETENTION_PYTHON set "RETENTION_PYTHON=%%P"
if defined RETENTION_PYTHON goto launch
where py.exe >nul 2>nul
if not errorlevel 1 goto pylauncher
echo Install 64-bit Python 3.12 or newer, then run this launcher again.
echo Or set RETENTION_PYTHON to your Python executable.
pause
exit /b 1
:launch
rem Setup needs a console interpreter even if an override names pythonw.
for %%P in ("%RETENTION_PYTHON%") do if /I "%%~nxP"=="pythonw.exe" set "RETENTION_PYTHON=%%~dpPpython.exe"
"%RETENTION_PYTHON%" "%~dp0bootstrap_runtime.py" %*
goto result
:pylauncher
py -3 "%~dp0bootstrap_runtime.py" %*
:result
if errorlevel 1 (
echo Startup failed. Details are in analysis\runtime-setup.log or analysis\launch-error.log.
if "%~1"=="" pause
exit /b 1
)
exit /b 0
