cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --HE --tag .HECB --separate true %%i
)

pause