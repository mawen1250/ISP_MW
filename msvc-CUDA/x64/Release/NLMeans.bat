cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --NLMeans --correction true --sigma 8.0 --BlockSize 8 --BlockStep 5 --GroupSize 16 --BMrange 24 --BMstep 3 %%i
)

pause