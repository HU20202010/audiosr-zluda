# AudioSR with ZLUDA on AMD GPUs

本项目旨在通过 **ZLUDA** 技术，使原本仅支持 NVIDIA CUDA 的 **AudioSR** (Audio Super-Resolution) 能够在 **AMD GPU** 上运行。

> ⚠️ **重要警告**：本项目处于**早期试验阶段**。不保证在所有设备上都能稳定运行，可能会遇到崩溃、性能低下或结果不一致的问题。请谨慎使用。

## 📋 前置要求

在开始之前，请确保你的系统满足以下条件：

1.  **操作系统**: Windows 10/11 (64-bit)
2.  **GPU**: AMD Radeon GPU (支持 ROCm/HIP)
3.  **Python**: Python 3.10
## 🛠️ 安装步骤

### 1. 安装 AMD ROCm HIP SDK

你需要安装 AMD 官方的 ROCm HIP SDK 以提供底层 GPU 支持。

*   **版本要求**: ROCm HIP SDK **7.2**
*   请前往 AMD 开发者官网下载并安装对应版本的 SDK。

### 2. 创建虚拟环境并安装 AudioSR

建议使用 `venv` 创建一个干净的 Python 环境。

请安装指定版本的cuda torch **https://download-r2.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp310-cp310-win_amd64.whl#sha256=af4833e36a8e964681a4dad7775f559cf043bd42c9d0c0b5e0619f9d0e44cb56**
***把项目代码覆盖到audiosr包文件！***

### 3. 安装zluda

确保你的zluda地址为
**"G:\zluda\zluda.exe"**

或通过其他方式修改

### 4. 下载模型

访问hugging face 下载**roberta-base**和**models--haoheliu--audiosr_basic** 放入  **C:\Users\<your_name>\.cache\huggingface\hub**
你也可以修改代码安装位置



