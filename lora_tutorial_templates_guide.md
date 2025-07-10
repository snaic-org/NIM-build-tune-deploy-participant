# LoRA Tutorial: Template Scripts vs Production Scripts

This guide explains the difference between the educational template scripts created in the tutorial and the production scripts used for actual training.

## 🎓 Educational Template Scripts

These scripts are created to demonstrate core concepts in a simplified, readable format:

### Template Scripts Created:
1. **`train_lora_template.py`**
   - Shows the basic flow of LoRA training
   - Demonstrates key APIs: loading models, adding adapters, training loop
   - ~70 lines of clear, commented code

2. **`inference_lora_template.py`**
   - Illustrates how to load a model with LoRA adapter
   - Shows basic text generation
   - ~60 lines focusing on core concepts

3. **`merge_lora_template.sh`**
   - Demonstrates the merge command
   - Explains what happens during merging
   - Simple bash script with educational comments

4. **`export_lora_template.py`**
   - Shows the two-step process: merge then export
   - Illustrates TensorRT export parameters
   - ~80 lines with clear function separation

## 🚀 Production Scripts (NeMo)

These are the actual scripts used for training and deployment:

### Production Scripts Used:
1. **Training**: `NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py`
   - 1000+ lines of production code
   - Handles distributed training, checkpointing, logging
   - Supports multiple model architectures
   - Advanced error handling and recovery

2. **Inference**: `NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py`
   - Batch processing for efficiency
   - Multiple generation strategies
   - Distributed inference support
   - Performance optimizations

3. **Merge**: `NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py`
   - Handles distributed checkpoints
   - Validates compatibility
   - Memory-efficient merging

4. **Export**: `NeMo/scripts/export/export_to_trt_llm.py`
   - GPU-specific optimizations
   - Multiple quantization options
   - Graph optimization
   - Performance profiling

## 🎯 When to Use Which?

### Use Templates For:
- **Learning**: Understanding the core concepts
- **Teaching**: Explaining LoRA to others
- **Prototyping**: Quick experiments with custom logic
- **Debugging**: Understanding what's happening

### Use Production Scripts For:
- **Real Training**: Actual model fine-tuning
- **Deployment**: Production systems
- **Performance**: When speed matters
- **Scale**: Large models or datasets

## 💡 Key Differences

| Aspect | Templates | Production |
|--------|-----------|------------|
| Lines of Code | 50-100 | 500-2000 |
| Error Handling | Basic | Comprehensive |
| Performance | Educational | Optimized |
| Features | Core only | Full suite |
| Dependencies | Minimal | Complete |
| Documentation | Inline teaching | API focused |

## 📝 Best Practice

1. **Start with templates** to understand the concepts
2. **Read template code** to see the flow
3. **Use production scripts** for actual work
4. **Refer back to templates** when debugging

The templates are your "Rosetta Stone" - they translate the complex production code into understandable concepts! 