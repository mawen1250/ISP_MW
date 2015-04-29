cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Retinex_MSRCP --sigma 15 --sigma 250 --lower_thr 0.01 --upper_thr 0.01 %%i
)

pause