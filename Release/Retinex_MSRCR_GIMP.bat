cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Retinex_MSRCR_GIMP --sigma 25 --sigma 80 --sigma 250 --dynamic 10 %%i
)

pause