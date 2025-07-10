# NIM Workshop - Presenter Edition

This repository contains workshop materials for learning about NVIDIA NIM (NVIDIA Inference Microservice) with Llama 3.2 1B Instruct model.

## Prerequisites

- NGC Account and API Key from [ngc.nvidia.com](https://ngc.nvidia.com)
- NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com)
- Docker installed and running
- 15GB+ free disk space
- (Optional) NVIDIA GPU for local deployment

## Quick Start

1. **Run the setup notebook**: `00_Workshop_Setup.ipynb`
   - Sets up API keys
   - Downloads Llama 3.2 1B Instruct model
   - Pulls NIM Docker container

2. **Follow the workshop notebooks in order**:
   - `01_NIM_API_Tutorial_with_scripts.ipynb` - Use cloud-hosted NIMs
   - `02_Local_NIM_Deployment_with_scripts.ipynb` - Deploy locally with Docker
   - `03_LoRA_Training_NeMo_with_scripts_FIXED.ipynb` - Fine-tune with LoRA
   - `04_Deploy_LoRA_with_NIM_with_scripts.ipynb` - Deploy fine-tuned models

## Model Information

- **Model**: Llama 3.2 1B Instruct
- **Size**: 2.3GB
- **Format**: NeMo 2 distributed checkpoint
- **NGC Path**: `nvidia/nemo/llama-3_2-1b-instruct:2.0`

## Workshop Contents

### 1. NIM API Tutorial
Learn to use NVIDIA's cloud-hosted inference endpoints for immediate AI capabilities.

### 2. Local NIM Deployment
Deploy NIMs on your own hardware using Docker containers.

### 3. LoRA Fine-tuning
Customize the model for your specific use case using Low-Rank Adaptation.

### 4. Deploy LoRA with NIM
Serve your fine-tuned models through the NIM framework.

## Troubleshooting

If you encounter issues:
1. Verify your API keys are correct
2. Ensure Docker is running
3. Check that you have sufficient disk space
4. For model downloads, ensure you're using NGC CLI from the `ngc-cli` directory

## Support

For questions or issues, please refer to:
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NGC Support](https://ngc.nvidia.com/support)

## Workshop Structure

```
NIM Workshop - Presenter/
├── 00_Workshop_Setup.ipynb
├── 01_NIM_API_Tutorial_with_scripts.ipynb
├── 02_Local_NIM_Deployment_with_scripts.ipynb
├── 03_LoRA_Training_NeMo_with_scripts_FIXED.ipynb
├── 04_Deploy_LoRA_with_NIM_with_scripts.ipynb
├── README.md
├── .gitignore
├── img/
│   └── sample_image.jpg
├── lora_tutorial/
│   ├── configs/
│   ├── data/
│   └── models/
└── ngc-cli/
```

## Common Issues

### NGC CLI Installation
The setup notebook handles NGC CLI installation automatically. The CLI is pre-installed in the `ngc-cli` directory.

### Docker Authentication
If Docker pulls fail with "unauthorized":
```bash
docker login nvcr.io -u $oauthtoken -p YOUR_NGC_API_KEY
```

### Model Download Issues
The Llama 3.2 1B Instruct model is publicly accessible with a standard NGC API key. If downloads fail:
1. Verify your NGC API key is correct
2. Check your internet connection
3. Ensure you have sufficient disk space (at least 3GB)

### Model Downloads
If automatic download fails, you can manually download from:
- [Llama 3.2 1B Instruct](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama-3_2-1b-instruct-nemo)

Place the `.nemo` file in `lora_tutorial/models/llama-3.2-1b-instruct-nemo/`

## For Workshop Presenters

- Run the setup notebook before the workshop to pre-download everything
- The `.env` file saves API keys between sessions
- Test Docker containers with:
  ```bash
  docker run --rm nvcr.io/nim/meta/llama-3.2-1b-instruct:latest echo "Ready!"
  ```

## Security Note

Never commit `.env` files to git! Add to `.gitignore`:
```
.env
*.nemo
__pycache__/
``` 