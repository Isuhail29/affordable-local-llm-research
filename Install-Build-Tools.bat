@echo off
echo ==========================================================
echo   Installing the compiler toolchain for the research
echo   (cmake + Visual Studio Build Tools + CUDA compiler)
echo.
echo   IMPORTANT: this window must be started with
echo   "Run as administrator" or the installs will fail.
echo.
echo   Total download is about 9 GB. This can take an hour.
echo   Leave this window open until it says ALL DONE.
echo ==========================================================
winget install --id Kitware.CMake --silent --accept-package-agreements --accept-source-agreements
winget install --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Nvidia.CUDA --silent --accept-package-agreements --accept-source-agreements
echo.
echo ==========================================================
echo   ALL DONE! Close this window and tell Claude to continue.
echo ==========================================================
pause
