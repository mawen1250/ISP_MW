cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Retinex_MSRCP --sigma 25 --sigma 80 --sigma 250 --lower_thr 0.001 --upper_thr 0.001 %%i
)

pause