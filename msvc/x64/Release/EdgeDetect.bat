cd /d "%~dp0"

FOR %%i IN (%*) DO (
    ISP_MW --EdgeDetect --kernel Sobel %%i
)

pause