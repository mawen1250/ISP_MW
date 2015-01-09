cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --NLMeans --sigma 8.0 --BlockSize 8 --Overlap 4 --BMrange 24 --BMstep 3 %%i
)

pause