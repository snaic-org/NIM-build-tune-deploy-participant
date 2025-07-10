# NeMo 1.x to 2.0 Compatibility Guide

## Current Situation (as of NeMo 2.0.0rc0)

### What We Have
- **NeMo Version**: 2.0.0rc0 (release candidate)
- **Checkpoint Format**: NeMo 2.0 (distributed .distcp format)
- **Training Scripts**: Still expect NeMo 1.0 structure

### The Compatibility Gap
```
NeMo 1.0 expects:  checkpoint_dir/model_config.yaml
NeMo 2.0 has:      checkpoint_dir/context/model.yaml
```

## Recommended Solution: Symlink

### Why Symlinks Are Correct Here
1. **Standard practice** for version compatibility
2. **Zero overhead** - no file duplication
3. **Transparent** - clearly shows the mapping
4. **Reversible** - easy to remove when no longer needed
5. **Safe** - doesn't modify original files

### How to Create the Symlink
```bash
cd /path/to/checkpoint
ln -s context/model.yaml model_config.yaml
```

### When to Use This
- ✅ Loading NeMo 2.0 checkpoints with current training scripts
- ✅ During the transition period (NeMo 2.0.0rc versions)
- ✅ When you need backward compatibility

### When NOT Needed
- ❌ Once NeMo 2.0 final is released with updated scripts
- ❌ If using pure NeMo 1.0 checkpoints
- ❌ If using future NeMo 2.0 APIs (when available)

## Future-Proofing Your Code

### Current Approach (Symlink)
```python
# Check if compatibility symlink is needed
import os
model_path = "path/to/checkpoint"
if os.path.exists(os.path.join(model_path, "context/model.yaml")) and \
   not os.path.exists(os.path.join(model_path, "model_config.yaml")):
    # Create symlink for compatibility
    os.chdir(model_path)
    os.symlink("context/model.yaml", "model_config.yaml")
```

### Future NeMo 2.0 API (when available)
```python
# This will work natively with NeMo 2.0 format
from nemo.collections import llm
llm.finetune(model="path/to/checkpoint", ...)
```

## Summary

Using symlinks is the **correct, professional approach** for handling this transition period. It's:
- Clean and transparent
- Following software engineering best practices
- Allowing work to continue without waiting
- Easy to remove when no longer needed

This is exactly how major software projects handle version transitions!
