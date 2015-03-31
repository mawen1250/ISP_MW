cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --HE --tag .HECB --strength 0.5 --separate true %%i
)

pause