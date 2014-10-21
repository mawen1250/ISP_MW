cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --HE --separate false %%i
)

pause