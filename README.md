# NIM Workshop - Presenter Materials

This folder contains all the materials needed to present the NVIDIA NIM workshop on building, tuning, and deploying LLMs.

## 📚 Workshop Notebooks

1. **00_Workshop_Setup.ipynb** - Initial setup and model download
2. **01_Introduction_to_NIMs.ipynb** - Understanding NVIDIA NIMs
3. **02_Running_NIM_Containers.ipynb** - Hands-on with NIM containers
4. **03_LoRA_Training_NeMo_with_scripts.ipynb** - Fine-tuning with LoRA
5. **04_Deploying_LoRA_NIMs.ipynb** - Deploying custom models

## 📁 Required Directories

- `lora_tutorial/` - Created during the workshop for LoRA training
- `nim_cache/` - For NIM container caching
- `ngc-cli/` - NGC CLI installation (included)
- `NeMo/` - Cloned during notebook 03

## 🚀 Getting Started

1. Ensure you have your NGC API key ready
2. Start with `00_Workshop_Setup.ipynb`
3. Follow the notebooks in order
4. Each notebook builds on the previous one

## 📖 Additional Documentation

- **NeMo_2_explanation.md** - Comprehensive guide to NeMo Framework & distributed checkpoint format
  - Explains the entire NeMo ecosystem (Framework, Guardrails, Curator, Aligner)
  - Details the `.distcp` file format you'll encounter
  - Compares NeMo with other frameworks (Hugging Face, PyTorch, DeepSpeed)
  - Includes extensive Q&A section covering common questions
  - When to use which NeMo component
  
- **nemo_ecosystem_diagram.mmd** - Visual diagram of the NeMo ecosystem
  - Shows relationships between NeMo components
  - Illustrates the flow from training to deployment

## 🔧 Requirements

- GPU with at least 24GB memory (RTX 4090 or better)
- Docker with GPU support
- ~50GB free disk space
- NGC account and API key

## 📝 Notes

### Technical Details

- All notebooks include presenter scripts for a seamless teaching experience
- The Llama 3.1 8B model is used throughout for efficiency
- Each notebook is self-contained with all necessary setup steps
- LoRA training takes approximately 5-10 minutes
- NIM deployment requires pulling container images (~15GB)

---

For the participant version without presenter scripts, see the "NIM Workshop - Participant" folder. 