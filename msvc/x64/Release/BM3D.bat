cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --BM3D --profile fast --sigma 10.0 %%i
)

pause

FOR %%i IN (%*) DO (
    rem ISP_MW --BM3D --profile fast --sigma 10.0 --sigma 10.0 --sigma 10.0 --BlockSize1 8 --BlockStep1 8 --GroupSize1 16 --BMrange1 9 --BMstep1 1 --lambda 2.7 --BlockSize2 8 --BlockStep2 7 --GroupSize2 16 --BMrange2 9 --BMstep2 1 %%i
)
