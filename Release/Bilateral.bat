cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Bilateral --sigmaS 3.0 --sigmaR 0.02 %%i
)

pause