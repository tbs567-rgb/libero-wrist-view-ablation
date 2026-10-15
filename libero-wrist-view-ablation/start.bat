@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d %~dp0

rem ============ Python 解释器检测（有 .venv 用 .venv，否则用系统 python）============
if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)

:menu
cls
echo ==================================================
echo    LIBERO 腕部视角实验系统（双视角消融平台）
echo ==================================================
echo    [1] 环境自检（python / mujoco / torch / 数据）
echo    [2] 训练模型（单相机 / 双相机 / 仅腕部）
echo    [3] 官方协议评测（50 局，256 渲染）
echo    [4] 机制探针（四阶段 + 失败标签）
echo    [5] 生成论文图表（fig1-fig9）
echo    [6] 数字现算复核（data_audit.py）
echo    [7] 查看已有结果汇总
echo    [0] 退出
echo ==================================================
set /p c=请选择:

if "%c%"=="1" goto check
if "%c%"=="2" goto train
if "%c%"=="3" goto eval
if "%c%"=="4" goto probe
if "%c%"=="5" goto figs
if "%c%"=="6" goto audit
if "%c%"=="7" goto results
if "%c%"=="0" exit /b
goto menu

rem ============ 各功能 ============
:check
%PY% -c "import sys; print('Python:', sys.version.split()[0])"
%PY% -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
%PY% -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
if exist "datasets\libero_spatial" (echo 数据: datasets\libero_spatial 存在) else (echo 数据: 未找到 datasets\libero_spatial —— 请先下载 LIBERO 数据集)
pause
goto menu

:train
set /p h5=演示数据 hdf5 路径（回车用默认任务）:
if "%h5%"=="" set h5=datasets\libero_spatial\pick_up_the_black_bowl_between_the_plate_and_the_ramekin_demo.hdf5
echo   [1] 仅全局（单相机）  [2] 双相机  [3] 仅腕部
set /p cam=相机配置:
if "%cam%"=="2" (set FLAG=--dual-cam) else if "%cam%"=="3" (set FLAG=--wrist-cam) else (set FLAG=)
set /p ep=训练轮数（回车 600）:
if "%ep%"=="" set ep=600
%PY% low_resource_official.py train --h5 "%h5%" --epochs %ep% %FLAG%
pause
goto menu

:eval
set /p h5=演示数据 hdf5 路径:
echo   [1] 仅全局  [2] 双相机  [3] 仅腕部
set /p cam=相机配置:
if "%cam%"=="2" (set FLAG=--dual-cam) else if "%cam%"=="3" (set FLAG=--wrist-cam) else (set FLAG=)
set /p tid=任务编号 task-id:
set LIBERO_EVAL_CAM=256
%PY% low_resource_official.py eval --h5 "%h5%" --task-id %tid% %FLAG%
pause
goto menu

:probe
echo 机制探针（默认跑全部配置，详见 stage_analysis.py）
%PY% stage_analysis.py
pause
goto menu

:figs
%PY% make_figs.py
%PY% make_figs_v2.py
echo 图表已输出到 figures\
pause
goto menu

:audit
%PY% data_audit.py
pause
goto menu

:results
echo ===== results\ 目录下的成绩 =====
for %%f in (results\*.json) do echo   %%~nxf
echo.
echo 最新三柱表（仅腕部对照）：
%PY% -c "import json;r=json.load(open('results/wrist_check_results.json',encoding='utf-8'));[print(' ',k,v['rate']) for k,v in r.items()]"
pause
goto menu
