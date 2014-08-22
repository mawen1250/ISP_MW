cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --sigmaS 1.0 --sigmaR 0.1 %%i
)

pause