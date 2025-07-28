# Memory-Efficient LLM Fine-tuning with 4-bit Quantization

An optimized implementation for fine-tuning Mistral-7B on insurance FAQ data using 4-bit quantization and LoRA, designed to run efficiently on budget GPUs like T4.

## 🎯 Project Overview

This project showcases how to fine-tune large language models in resource-constrained environments using advanced quantization techniques. By leveraging 4-bit precision and LoRA adapters, we achieve effective model customization with significantly reduced memory requirements.

**Key Achievement**: Successfully fine-tune a 7B parameter model on a T4 GPU (15GB) with only 8GB memory usage.

## ✨ Highlights

- **🔧 4-bit Quantization**: Reduce memory usage by 50% using BitsAndBytes
- **⚡ T4 GPU Compatible**: Run on budget-friendly Google Colab T4 instances  
- **🎯 Domain Specialization**: Insurance FAQ question-answering optimization
- **📊 Performance Tracking**: Comprehensive before/after evaluation
- **⏱️ Fast Training**: Complete fine-tuning in under 7 minutes

## 🛠️ Technical Stack

### Model Architecture
```
Base Model: mistralai/Mistral-7B-Instruct-v0.2
├── Quantization: 4-bit precision (BitsAndBytes)
├── Effective Size: ~3.7B parameters
├── LoRA Adapters: 41.9M trainable parameters (1.11%)
└── Memory Footprint: ~8GB VRAM
```

### Hardware Requirements
- **Minimum**: T4 GPU (15GB VRAM) 
- **Recommended**: Any GPU with 8GB+ VRAM
- **RAM**: 12GB+ system memory
- **Storage**: 20GB free space

## 📦 Installation

```bash
# Core dependencies
pip install torch transformers datasets
pip install bitsandbytes peft accelerate
pip install pandas numpy huggingface_hub wandb
```

## 🚀 Quick Start Guide

### Step 1: Environment Setup
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Verify CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
```

### Step 2: Data Preparation
Create `data/faq_sample.json`:
```json
[
  {
    "question": "What is comprehensive auto insurance?",
    "answer": "Comprehensive coverage protects against non-collision damage like theft, vandalism, and weather damage."
  }
]
```

### Step 3: Authentication Setup
```python
from huggingface_hub import login
login(token="your_huggingface_token")
```

### Step 4: Run Training
```python
# Execute the main training pipeline
run_evaluation()
```

## ⚙️ Configuration Details

### Quantization Configuration
```python
# 4-bit model loading
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # Key memory optimization
    trust_remote_code=True
)
```

### LoRA Settings
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # Rank (complexity vs efficiency)
    lora_alpha=32,     # Scaling factor
    lora_dropout=0.1,  # Regularization
    target_modules=[   # Transformer layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Training Hyperparameters
```python
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision for speed
    optim="adamw_torch",
    lr_scheduler_type="cosine"
)
```

## 📈 Performance Results

### Training Metrics
| Metric | Value |
|--------|-------|
| **Initial Loss** | 3.26 |
| **Final Loss** | 0.075 |
| **Training Time** | 6 minutes 42 seconds |
| **GPU Utilization** | T4 (15GB VRAM) |
| **Memory Usage** | ~8GB peak |
| **Epochs** | 5 |

### Memory Efficiency Comparison
| Configuration | VRAM Usage | Speedup |
|---------------|------------|---------|
| Full Precision (FP32) | ~28GB | 1x |
| Half Precision (FP16) | ~14GB | 1.8x |
| **4-bit Quantized** | **~8GB** | **3.5x** |

### Quality Assessment

**Sample Question**: "How do I start a property damage claim?"

**Before Fine-tuning**:
> To start a property damage claim, follow these steps: 1. Document the Damage: Take clear photographs or videos...

**After Fine-tuning**:
> To start a property damage claim, contact your insurance company immediately, document the damage with photos, obtain a police report if applicable, keep receipts for temporary repairs, and cooperate with the claims adjuster during the investigation process.

**Improvement**: ✅ More concise, actionable, and insurance-specific

## 🔍 Architecture Deep Dive

### Memory Optimization Strategy
1. **4-bit Quantization**: Weights stored in 4-bit precision
2. **LoRA Adapters**: Only train small adapter matrices
3. **Gradient Checkpointing**: Trade compute for memory
4. **Mixed Precision**: FP16 for forward/backward passes