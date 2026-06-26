@echo off
REM =======================================================================================
REM Lancement sans Docker - Windows
REM Cree un environnement virtuel Python isole, installe les dependances si besoin,
REM puis lance l'application Streamlit.
REM =======================================================================================

cd /d "%~dp0"

set VENV_DIR=.venv

REM Verifie que Python est disponible
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python n'est pas trouve dans le PATH. Installez-le depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Cree le venv s'il n'existe pas encore
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creation de l'environnement virtuel...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

REM Installe/met a jour les dependances seulement si requirements.txt a change
set HASH_FILE=%VENV_DIR%\requirements.hash
for /f %%H in ('certutil -hashfile requirements.txt SHA256 ^| findstr /v "hash"') do set CURRENT_HASH=%%H

set OLD_HASH=
if exist "%HASH_FILE%" set /p OLD_HASH=<"%HASH_FILE%"

if not "%OLD_HASH%"=="%CURRENT_HASH%" (
    echo Installation des dependances ^(peut prendre quelques minutes la premiere fois^)...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    echo %CURRENT_HASH% > "%HASH_FILE%"
) else (
    echo Dependances deja installees et a jour.
)

set PYVISTA_OFF_SCREEN=true

echo Lancement de l'application sur http://localhost:8501
streamlit run app.py

pause
