cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --AWB1 %%i
)

pause