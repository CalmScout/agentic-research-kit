@echo off
REM Start Phoenix observability server for multi-agent RAG tracing (Windows)

echo 🔥 Starting Phoenix Observability Server...
echo.

REM Default port
set PORT=%PHOENIX_PORT%
if "%PHOENIX_PORT%"=="" set PORT=6006

REM Start Phoenix
echo 🚀 Starting Phoenix on port %PORT%...
echo    UI will be available at: http://localhost:%PORT%
echo.
echo Press Ctrl+C to stop Phoenix
echo.

python -m phoenix.server.main serve --port %PORT%

REM If stopped, cleanup
:end
echo 🛑 Phoenix stopped
