cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --sigma 1.0 %%i
)

pause