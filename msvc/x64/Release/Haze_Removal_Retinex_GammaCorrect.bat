cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --Haze_Removal_Retinex --TransferChar 1 --Ymode 1 --sigma 15 --sigma 250 --tMap_thr 0.001 --ALmax 1 --tMapMin 0.1 --tMapMax 1.2 --strength 0.85 --ppmode 3 --pp_sigma 10 --lower_thr 0.05 --upper_thr 0.03 --HistBins 1024 --debug 0 --tag .Haze_Removal_Retinex_GammaCorrect %%i
)

pause