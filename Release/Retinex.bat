cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Retinex --sigma 25 --sigma 80 --sigma 250 --lower_thr 0 --upper_thr 0 %%i
)

pause