cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Gaussian --sigma 3.0 %%i
)

pause