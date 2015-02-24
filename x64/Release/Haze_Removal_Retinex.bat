cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Haze_Removal_Retinex --Ymode 1 --sigma 25 --sigma 250 --tMap_thr 0.001 --ALmax 1 --tMapMin 0.1 --strength 0.65 --ppmode 3 --lower_thr 0.02 --upper_thr 0.01 --HistBins 1024 %%i
)

pause