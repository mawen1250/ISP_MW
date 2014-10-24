cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --HE --strength 0.5 --separate false %%i
)

pause