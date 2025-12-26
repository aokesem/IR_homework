@echo off
chcp 65001 > nul


echo ====================================
echo RAG问答系统 启动脚本 (本地模型)
echo ====================================
echo.

echo 使用本地模型: Qwen/Qwen3-0.6B
echo 模型缓存位置: C:\Users\aokesem\.cache\huggingface\hub
echo.

echo 检查Python环境...
D:\Anaconda3\envs\LLM_env\python.exe --version
if errorlevel 1 (
    echo 错误: Python环境未找到
    pause
    exit /b 1
)

echo.
echo 启动RAG系统...
echo 正在加载本地模型,请稍候...
echo.

D:\Anaconda3\envs\LLM_env\python.exe app.py

pause
