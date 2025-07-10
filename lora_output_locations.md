# LoRA Training Output Locations Guide

This document explains where all output files are saved when running the NeMo scripts in notebook 03.

## 🔄 Training Outputs (`megatron_gpt_finetuning.py`)

When you run the LoRA training script, outputs are saved to:

### Primary Location: `lora_tutorial/experiments/customer_support_lora/`
- **checkpoints/**
  - `customer_support_lora.nemo` (21MB) - The exported LoRA adapter ready for deployment
  - `megatron_gpt_peft_lora_tuning--validation_loss=X.XXX-step=XX-consumed_samples=XXX.ckpt` - Training checkpoints with optimizer state
- **logs/** - Training logs, loss curves, and TensorBoard files
- **config.yaml** - Complete training configuration

## 📊 Inference Outputs (`megatron_gpt_generate.py`)

When you run inference with the LoRA adapter:

### Location: Current working directory (workspace root)
- `customer_support_lora_test_customer_support_inputs_preds_labels.jsonl` - Contains:
  - `input`: The prompt given to the model
  - `pred`: Model's generated response
  - `label`: Expected response from test data

Example:
```json
{
  "input": "User: My package is damaged. What should I do?\n\nAssistant:",
  "pred": " I'm sorry to hear you're experiencing issues...",
  "label": " I'm sorry to hear you received a damaged product..."
}
```

## 🔧 Merge Outputs (`merge_lora_weights/merge.py`)

When merging LoRA adapter with base model:

### Location: `lora_tutorial/models/`
- `llama3_2-1b-customer-support-merged.nemo` - Full model with LoRA weights permanently merged

## 📝 Important Notes

1. **Experiment Tracking**: Each training run creates a new subdirectory in `experiments/` with timestamps
2. **Checkpoint Management**: Keep the `.nemo` file for deployment, archive `.ckpt` files for resuming training
3. **Disk Space**: Training outputs can consume several GB - monitor available space
4. **Clean Up**: After successful training, you can delete `.ckpt` files to save space

## 🚀 Production Deployment

For NIM deployment, you only need:
- `customer_support_lora.nemo` (21MB) - The LoRA adapter
- Base model files (unchanged)

The adapter can be dynamically loaded at inference time without merging. 