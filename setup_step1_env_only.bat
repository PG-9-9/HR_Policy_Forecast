@echo off
REM Step-by-step environment setup for policy_api (Windows)
echo Step 1: Creating policy_api environment...

REM Just create the environment first
conda create -n policy_api python=3.10 -y

echo.
echo Step 2: Check if environment was created...
conda env list | findstr policy_api

echo.
echo Environment creation complete!
echo.
echo Next steps:
echo 1. conda activate policy_api
echo 2. Run the package installer: python check_and_install_deps.py
echo.
pause