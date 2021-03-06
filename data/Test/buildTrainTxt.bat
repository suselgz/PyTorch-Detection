@echo off
setlocal DISABLEDELAYEDEXPANSION
setlocal ENABLEDELAYEDEXPANSION
del Test.txt 2>nul
for %%i in (mao shu xiong) do (
	for %%j in ( Test ) do (
		for /f %%l in ('dir /b %%i%%j\*.jpg') do echo %%i%%j\%%~nl>>Test.txt
	)
)
pause