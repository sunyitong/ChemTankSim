@echo off
REM Build pbrt-v4 (CPU) with MSVC 2022 + Ninja. Run from anywhere.
REM GPU/OptiX: install OptiX SDK 8.x/9.x, then add -DPBRT_OPTIX7_PATH="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0"
setlocal
set ROOT=%~dp0
set NINJA=%ROOT%liquid_level_sim\.venv\Scripts\ninja.exe
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist "%ROOT%pbrt-v4\build" mkdir "%ROOT%pbrt-v4\build"
cd /d "%ROOT%pbrt-v4\build"
cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DOPENEXR_FORCE_INTERNAL_IMATH=ON %*
if errorlevel 1 exit /b 1
"%NINJA%" pbrt_exe imgtool
if errorlevel 1 exit /b 1
echo BUILD_OK
dir /b *.exe
