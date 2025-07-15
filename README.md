# NIM Build, Tune & Deploy Workshop

This repository contains hands-on notebooks for the NVIDIA NIM workshop, demonstrating how to build, fine-tune, and deploy Large Language Models using NVIDIA's AI stack.

## 🎯 Workshop Overview

Learn how to:
- Use NVIDIA NIM APIs for cloud-based inference
- Deploy NIM containers locally with GPU acceleration
- Fine-tune LLMs using LoRA (Low-Rank Adaptation) with NeMo
- Deploy custom LoRA adapters with NIM for production use

## 📚 Workshop Notebooks

1. **[00_Workshop_Setup.ipynb](00_Workshop_Setup.ipynb)** - Initial setup and environment configuration
2. **[01_NIM_API_Tutorial_with_scripts.ipynb](01_NIM_API_Tutorial_with_scripts.ipynb)** - Introduction to NVIDIA NIM cloud APIs
3. **[02_Local_NIM_Deployment_with_scripts.ipynb](02_Local_NIM_Deployment_with_scripts.ipynb)** - Deploy NIM containers locally
4. **[03_LoRA_Training_NeMo_with_scripts.ipynb](03_LoRA_Training_NeMo_with_scripts.ipynb)** - Fine-tune models with LoRA
5. **[04_Deploy_LoRA_with_NIM_with_scripts.ipynb](04_Deploy_LoRA_with_NIM_with_scripts.ipynb)** - Deploy LoRA adapters

## 🚀 Prerequisites

- NVIDIA GPU (A100, V100, or similar)
- Docker with NVIDIA nvcr.io/nvidia/nemo:24.05.01 Container Runtime
- Python 3.8+
- NGC Account (free at [ngc.nvidia.com](https://ngc.nvidia.com))

## 🛠️ Quick Start

1. Clone this repository:
```bash
git clone https://github.com/darren236/NIM-build-tune-deploy-participant.git
cd NIM-build-tune-deploy-participant
```

2. Run the setup notebook:
```bash
jupyter notebook 00_Workshop_Setup.ipynb
```

3. Follow the notebooks in order (00 → 01 → 02 → 03 → 04)

## 📁 Repository Structure

```
NIM-build-tune-deploy-participant/
├── 00_Workshop_Setup.ipynb              # Environment setup
├── 01_NIM_API_Tutorial_with_scripts.ipynb    # Cloud API tutorial
├── 02_Local_NIM_Deployment_with_scripts.ipynb # Local deployment
├── 03_LoRA_Training_NeMo_with_scripts.ipynb  # LoRA training
├── 04_Deploy_LoRA_with_NIM_with_scripts.ipynb # LoRA deployment
├── lora_tutorial/                       # Training data and configs
│   └── data/                           # Sample datasets
└── img/                                # Workshop images
```

## 🔧 Key Technologies

- **NVIDIA NIM**: Inference microservices for optimized model deployment
- **NeMo Framework**: For training and fine-tuning LLMs
- **LoRA**: Efficient fine-tuning technique
- **Docker**: Container-based deployment

## 📝 Notes

- The workshop uses Llama 3.1 8B Instruct as the base model

## 📄 License

This workshop material is provided for educational purposes. Model usage is subject to respective model licenses. 
