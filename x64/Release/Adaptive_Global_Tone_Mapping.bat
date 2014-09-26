cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --AGTM %%i
)

pause