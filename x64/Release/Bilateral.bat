cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Bilateral --sigmaS 2.5 --sigmaR 0.1 %%i
)

pause