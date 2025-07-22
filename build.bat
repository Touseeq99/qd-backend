@echo off
REM HR Assistant Docker Build Script for Windows
REM This script optimizes Docker builds for faster development

echo 🚀 Building HR Assistant with optimized caching...

REM Enable Docker BuildKit for faster builds
set DOCKER_BUILDKIT=1

REM Build with cache from previous builds
docker build ^
    --cache-from qd_hr_assistant-hr-assistant ^
    --tag qd_hr_assistant-hr-assistant ^
    --target production ^
    .
 
if %ERRORLEVEL% EQU 0 (
    echo ✅ Build completed successfully!
    
    echo 📊 Build information:
    docker images qd_hr_assistant-hr-assistant --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo.
    echo 🎯 To start the application:
    echo    docker-compose up -d
    echo.
    echo 🎯 To view logs:
    echo    docker-compose logs -f
) else (
    echo ❌ Build failed!
    exit /b 1
) 