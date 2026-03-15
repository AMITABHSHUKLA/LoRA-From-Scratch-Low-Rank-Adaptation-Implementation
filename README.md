# LoRA From Scratch — Low Rank Adaptation Implementation

## Overview

Large Language Models have demonstrated remarkable capabilities across a wide range of tasks. However, fine-tuning these models remains computationally expensive due to the massive number of parameters involved. Low Rank Adaptation (LoRA) addresses this challenge by enabling efficient fine-tuning through low-rank updates applied to existing model weights rather than retraining the full parameter set.

This repository contains a clean, from-scratch implementation of LoRA built to deeply understand the underlying mechanics of parameter-efficient fine-tuning. Instead of relying on external libraries such as PEFT, this implementation reconstructs the complete LoRA workflow including adapter injection, weight freezing, and weight merging.

The goal of this project is not just usage but internal understanding of how LoRA works at a systems and mathematical level.

---

## What is LoRA

LoRA is a parameter-efficient fine-tuning technique introduced by Microsoft Research that enables training large models by decomposing weight updates into low-rank matrices.

Instead of updating the full weight matrix:

```
W' = W + ΔW
```

LoRA represents the update as a low-rank decomposition:

```
ΔW = BA
```

Where

```
A ∈ R(r × d)
B ∈ R(k × r)
r << d
```

This allows training only a small number of parameters while keeping the original weights frozen.

The final computation becomes

```
W' = W + α/r * BA
```

This dramatically reduces

- memory usage
- training compute
- GPU requirements

while preserving strong performance.

---

## Fine Tuning vs LoRA

![LoRA Fine Tuning Comparison]([https://www.intel.com/content/dam/developer/articles/llm/fine-tuning-llama2-70b-and-lora-on-gaudi2/fine-tuning.png](https://www.intel.com/content/dam/developer/articles/llm/fine-tuning-llama2-70b-and-lora-on-gaudi2/fine-tuning.png))

Traditional fine-tuning updates every parameter in the model. This requires large GPU memory and significant computational resources.

LoRA changes this paradigm by

- freezing pretrained weights
- learning only low-rank matrices
- injecting updates through matrix decomposition

This allows extremely large models to be adapted using only a small number of additional parameters.

---

## Why LoRA Matters

Modern transformer models can contain billions of parameters, making full fine-tuning expensive and often impractical.

LoRA provides several advantages

- Train less than 1 percent of model parameters
- Freeze the base model for stability
- Reduce GPU memory footprint
- Maintain performance close to full fine-tuning
- Enable rapid experimentation across tasks

This technique has become the foundation of many modern systems including

- instruction tuning
- domain adaptation
- LLM personalization
- multimodal model adaptation

---

## Project Goal

This project rebuilds LoRA from the ground up to understand the internal mechanics typically abstracted away by high-level libraries.

The implementation focuses on

- mathematical clarity
- modular adapter architecture
- explicit weight injection
- minimal abstractions

The repository demonstrates how LoRA can be implemented with pure PyTorch primitives while maintaining flexibility similar to production libraries.

---

## Core Components Implemented

### LoRA Base Layer

A shared abstraction responsible for

- storing LoRA hyperparameters
- computing scaling factors
- managing dropout
- loading pretrained weights

This layer acts as the foundation for all LoRA modules.

---

### LoRA Linear Layer

The linear adapter injects low-rank updates into existing linear layers.

Original transformation

```
y = xW^T + b
```

With LoRA

```
y = xW^T + x(BA)^T * scaling + b
```

Where

- A and B are trainable low-rank matrices
- W remains frozen

This design allows efficient gradient updates without modifying base parameters.

---

### LoRA Embedding Layer

The embedding adapter applies LoRA to embedding matrices.

Instead of retraining the full embedding table, the implementation learns

```
Embedding = BaseEmbedding + LowRankUpdate
```

This enables parameter-efficient adaptation even for large vocabulary embeddings.

---

### LoRA Model Wrapper

A recursive module traversal system is implemented to

- scan the entire model
- identify target modules
- replace them with LoRA-enhanced layers

This dynamic injection allows LoRA to be applied without modifying the original model architecture.

Key responsibilities include

- parameter freezing
- adapter insertion
- bias training configuration
- adapter merging for inference

---

### Weight Merging for Inference

For deployment scenarios, LoRA updates can be merged into the base model weights.

```
W_final = W + BA
```

This allows inference without any adapter overhead.

---

## Repository Structure

```
LoRA_From_Scratch/

LoRA_BASE.py
Base LoRA abstraction and scaling logic

LORA_Linear.py
LoRA implementation for Linear layers

LORA_Embedding.py
LoRA implementation for Embedding layers

LORA_Model.py
Model wrapper that injects LoRA adapters

save_lora_weights.py
Utility for saving merged LoRA weights
```

---

## Example Workflow

```python
config = LORAConfig(
    rank=8,
    lora_alpha=8,
    lora_dropout=0.05,
    target_layers=["q_proj", "v_proj"]
)

model = LoraModel(base_model, config)
```

Training updates only the LoRA parameters while the base model remains frozen.

---

## What This Implementation Demonstrates

This project provides a transparent view of how LoRA actually works internally.

Key insights include

- how low-rank adapters integrate into neural networks
- how parameter freezing is handled
- how dynamic module replacement works in PyTorch
- how LoRA weights can be merged for inference

Understanding these mechanics is critical for designing custom parameter-efficient architectures beyond LoRA.

---

## Reference

Hu et al.  
LoRA: Low-Rank Adaptation of Large Language Models  

https://arxiv.org/abs/2106.09685

---

## Closing Notes

This repository is designed as a learning-focused reconstruction of LoRA aimed at deeply understanding the mechanics of parameter-efficient fine-tuning.

While modern frameworks provide convenient abstractions, rebuilding these systems from scratch offers invaluable insight into how large-scale model adaptation actually works under the hood.

Contributions and discussions are welcome.
