# Day 1 Master Plan: Tensors, Autograd, and the Mental Model

Welcome to Day 1. Since you already understand Machine Learning conceptually, today is entirely about translating that mathematical intuition into PyTorch's execution model. 

By the end of today, you will view PyTorch not as a rigid deep learning tool, but as an **elastic, GPU-accelerated mathematical sandbox that handles calculus automatically.**

---

## 📅 The Hourly Core Schedule
Treat today as an intense 4-hour focused block (or split into two 2-hour sessions).

| Block | Focus Area | Core Objective | Time |
| :--- | :--- | :--- | :--- |
| **Block 1** | **The Physics of Tensors** | Stop treating tensors like lists. Master memory layout and shapes. | 60 mins |
| **Block 2** | **The Computation Graph & Autograd** | Reverse-engineer how PyTorch secretly tracks operations. | 60 mins |
| **Block 3** | **Hardware Mechanics (`cpu` vs `cuda`)** | Master explicit cross-device memory transfers safely. | 30 mins |
| **Block 4** | **The Bare-Metal Milestone** | Build a linear regression engine from scratch without `nn`. | 90 mins |

---

## 🧠 Core Intuitions & Mental Models

### 1. Tensors vs. NumPy Arrays
A PyTorch Tensor is fundamentally a NumPy array with **two superpowers**:
1. **Device Awareness:** It can be cast directly onto specialized hardware (Nvidia CUDA GPUs, Apple Silicon MPS) to execute linear algebra thousands of times faster than a CPU.
2. **The Graph Hook:** It can carry a hidden metadata tag (`requires_grad=True`) that converts the tensor from a static chunk of numbers into an active node inside an automated calculus tracker.

### 2. The Dynamic Computation Graph (Autograd)
In static frameworks (like old TensorFlow), you built a pipeline first, then poured data through it. PyTorch uses a **DAG (Directed Acyclic Graph)** built *on the fly*. 

Every time you perform an operation on a tensor (`Z = X * W + B`), PyTorch creates a temporary tracking node. 
* The inputs point forward to the output.
* The output keeps a backward pointer (`grad_fn`) to the operation that created it.
* When you call `loss.backward()`, PyTorch walks in reverse from the final error score back to your original weights, executing the chain rule automatically.

---

## 🛠️ What You Must Master Today (The 80/20 Elements)

### 1. Shape Manipulation (Where 90% of bugs happen)
You must comfortably use these three methods without looking up documentation:
* `.view(*shape)` / `.reshape(*shape)`: Changes the structural layout of data without moving it in memory. Understand that `.view()` forces contiguous memory layout and is highly efficient.
* `.unsqueeze(dim)`: Adds a dimension of size 1 at the specified axis. Essential for transforming a single data instance into a "batch of 1" before feeding it to a model.
* `.squeeze(dim)`: Drops dimensions of size 1. Essential for converting output matrices back into simple 1D arrays or scalars.

### 2. Autograd Mechanics
* `requires_grad=True`: Tells PyTorch to allocate memory for a parallel tensor of the exact same shape called `.grad`.
* `loss.backward()`: The trigger that runs the calculus engine.
* `.detach()` or `with torch.no_grad()`: Removes a tensor from the tracking graph. Crucial for custom evaluation metrics or when updating weights manually.

---

## 🎯 The Bare-Metal Milestone: Capstone Project

**Objective:** Write a complete Linear Regression optimization sequence using *only* raw math. **Strictly banned:** `torch.nn`, `torch.optim`.

### The Step-by-Step Execution Blueprint

```python
import torch

# 1. Synthesize Synthetic Ground Truth Data
# Target Function: y = 3.5 * x + 2.0 + noise
X = torch.randn(100, 1)
noise = torch.randn(100, 1) * 0.1
Y = 3.5 * X + 2.0 + noise

# 2. Initialize Model Parameters (The weights we want to learn)
# We initialize them randomly, but turn on the calculus engine tracking
W = torch.randn(1, 1, requires_grad=True)
B = torch.zeros(1, requires_grad=True)

learning_rate = 0.1

# 3. The Optimization Loop
for epoch in range(100):
    # ---- Forward Pass ----
    # Compute predictions using standard matrix math
    predictions = X @ W + B 
    
    # Compute Mean Squared Error (MSE) Loss
    loss = torch.mean((predictions - Y) ** 2)
    
    # ---- Backward Pass ----
    # Compute gradients across the entire tracking family tree
    loss.backward()
    
    # ---- Weight Update Step ----
    # Wrap in no_grad because modifying weights shouldn't be tracked as part of the model math
    with torch.no_grad():
        W -= learning_rate * W.grad
        B -= learning_rate * B.grad
        
        # CRITICAL: PyTorch accumulates gradients. Clear them manually for the next loop!
        W.grad.zero_()
        B.grad.zero_()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f} | W = {W.item():.2f}, B = {B.item():.2f}")

print(f"\nFinal Optimized Weights -> W: {W.item():.4f}, B: {B.item():.4f}")