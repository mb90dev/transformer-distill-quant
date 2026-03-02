@echo off
setlocal

set "PROJECT_DIR=C:\Users\DELL\Desktop\GitHub\transformer-distill-quant"
set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
set "CONFIG_PATH=configs/base_cli.yaml"
set "INTERACTIVE=0"

if not exist "%PYTHON_EXE%" (
  echo [BLAD] Nie znaleziono pythona: %PYTHON_EXE%
  echo Ustaw poprawna sciezke PYTHON_EXE w tym pliku .bat
  echo.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
  echo [BLAD] Nie mozna wejsc do katalogu projektu: %PROJECT_DIR%
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  set "INTERACTIVE=1"
  goto MENU
)
set "COMMAND=%~1"
goto RUN

:MENU
echo.
echo === Transformer Distill Quant - Launcher ===
echo 1. prepare-data
echo 2. train-baseline
echo 3. train-teacher
echo 4. distill
echo 5. quantize
echo 6. benchmark
echo Q. wyjscie
echo.
choice /c 123456Q /n /m "Wybierz opcje: "
if errorlevel 7 goto END
if errorlevel 6 set "COMMAND=benchmark" & goto RUN
if errorlevel 5 set "COMMAND=quantize" & goto RUN
if errorlevel 4 set "COMMAND=distill" & goto RUN
if errorlevel 3 set "COMMAND=train-teacher" & goto RUN
if errorlevel 2 set "COMMAND=train-baseline" & goto RUN
if errorlevel 1 set "COMMAND=prepare-data" & goto RUN

:RUN
echo.
echo Uruchamiam: %COMMAND%
"%PYTHON_EXE%" -m src.cli --config "%CONFIG_PATH%" "%COMMAND%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo [BLAD] Komenda zakonczyla sie kodem %EXIT_CODE%.
) else (
  echo [OK] Komenda wykonana poprawnie.
)
echo.
pause
if "%INTERACTIVE%"=="1" goto MENU
exit /b %EXIT_CODE%

:END
echo Koniec.
exit /b 0
