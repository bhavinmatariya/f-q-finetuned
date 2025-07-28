# LLM Fine-tuning with LoRA for Insurance FAQ

A comprehensive implementation for fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) technique, specifically designed for insurance domain question-answering tasks.

## 🚀 Overview

This project demonstrates how to fine-tune the Mistral-7B-Instruct model on insurance FAQ data using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The implementation includes before/after comparison evaluation to demonstrate the effectiveness of fine-tuning.

## 🔧 Features

- **LoRA Integration**: Efficient fine-tuning using Low-Rank Adaptation
- **Insurance Domain**: Specialized for insurance FAQ and customer support
- **Comparison Analysis**: Before/after fine-tuning response comparison
- **GPU Optimized**: Designed for Google Colab with A100 GPU
- **Comprehensive Logging**: Detailed training progress and evaluation metrics

## 📋 Requirements

### Hardware
- GPU with at least 16GB VRAM (A100 recommended)
- 32GB+ RAM recommended

### Dependencies
```bash
pip install bitsandbytes transformers datasets peft torch pandas numpy huggingface_hub
```

### Environment
- Python 3.8+
- CUDA compatible GPU
- Google Colab (recommended) or local GPU setup

## 🗂️ Project Structure

```
├── FineTune_LLM_GPU.ipynb    # Main notebook
├── data/
│   └── faq_sample.json       # Insurance FAQ dataset
├── results/                  # Output directory
│   ├── comparison_results.txt
└── README.md
```

## 📊 Model Configuration

### Base Model
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Parameters**: 7.2B total parameters
- **Architecture**: Transformer-based causal language model

### LoRA Configuration
```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                    # Rank
    lora_alpha=32,          # Alpha parameter
    lora_dropout=0.1,       # Dropout rate
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none"
)
```

### Training Parameters
- **Trainable Parameters**: 41,943,040 (0.58% of total)
- **Epochs**: 5
- **Batch Size**: 1 (with gradient accumulation steps: 4)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine

## 📚 Dataset Format

The training data should be in JSON format with question-answer pairs:

```json
[
    {
        "question": "What does comprehensive coverage include?",
        "answer": "Comprehensive coverage includes all standard coverage options plus additional features..."
    },
    {
        "question": "How do I start a property damage claim?",
        "answer": "To start a property damage claim, contact your insurance company immediately..."
    }
]
```

## 🚀 Quick Start

### 1. Prepare Data
Place your insurance FAQ dataset in `data/faq_sample.json` following the format above.

### 2. Configure Authentication

#### Hugging Face Token
```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

#### Weights & Biases API Key
The project uses Weights & Biases for experiment tracking. You'll need to authenticate:

```python
# Option 1: During runtime (prompted in notebook)
# You'll be prompted to enter your API key when training starts
```

**Get your W&B API key**: 
1. Visit [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key
3. Paste it when prompted during training

### 4. Run Fine-tuning
Execute the Jupyter notebook `FineTune_LLM_GPU.ipynb` or run the main training pipeline:

```python
python run_evaluation()
```

## 📈 Training Results

### Performance Metrics
- **Initial Loss**: 3.35
- **Final Loss**: 0.08
- **Training Time**: ~2.5 minutes for 5 epochs
- **Memory Usage**: Optimized with mixed precision (FP16)

### Before vs After Comparison

**Question**: "What does comprehensive coverage include?"

**Before Fine-tuning**:
> Comprehensive coverage, also known as "full coverage," is a type of auto insurance that goes beyond the basic liability coverage required by law...

**After Fine-tuning**:
> Comprehensive coverage includes all standard coverage options plus additional features like roadside assistance, rental car reimbursement, and theft or vandalism coverage. It typically provides the most protection and peace of mind, but may cost more than basic coverage.

## 🔍 Key Components

### 1. Data Processing
- Converts FAQ data to instruction-following format
- Tokenizes using Mistral tokenizer
- Handles padding and truncation

### 2. Model Setup
- Loads pre-trained Mistral-7B-Instruct model
- Applies LoRA configuration
- Sets up mixed precision training

### 3. Training Pipeline
- Custom trainer with language modeling objective
- Gradient accumulation for memory efficiency
- Cosine learning rate scheduling

### 4. Evaluation
- Generates responses for test questions
- Compares before/after fine-tuning results
- Saves detailed comparison report

## 🛠️ Advanced Configuration

### Memory Optimization
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
training_args.fp16 = True

# Optimize batch size and accumulation
training_args.per_device_train_batch_size = 1
training_args.gradient_accumulation_steps = 4
```

## 📊 Monitoring and Logging

The project integrates with Weights & Biases for experiment tracking:

- Training loss progression
- Learning rate scheduling
- Memory usage monitoring
- Model performance metrics

### Performance Tips

- Use A100 GPU for optimal performance
- Enable mixed precision training
- Optimize data loading with multiple workers
- Monitor GPU memory usages