#!/usr/bin/env python3
"""
LoRA Training Script for NeMo
Usage: python train_lora.py
"""

import os
import torch
from omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.peft_config import LoraPEFTConfig
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

def main():
    # Load config
    cfg = OmegaConf.load("lora_tutorial/configs/lora_config.yaml")
    
    # Initialize trainer
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    
    # Setup experiment manager
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Load base model and merge configs
    model_cfg = MegatronGPTSFTModel.merge_cfg_with(
        cfg.model.restore_from_path, 
        cfg
    )
    
    # Initialize model
    model = MegatronGPTSFTModel.restore_from(
        cfg.model.restore_from_path, 
        model_cfg, 
        trainer=trainer
    )
    
    # Add LoRA adapter
    logging.info("Adding LoRA adapter...")
    model.add_adapter(LoraPEFTConfig(model_cfg))
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Start training
    logging.info("Starting LoRA training...")
    trainer.fit(model)
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main()
