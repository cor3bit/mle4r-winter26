# Machine Learning Experiments for Researchers

Practical companion to the [IMT Machine Learning course](https://cse.lab.imtlucca.it/~bemporad/ml.html).

This course teaches how to design, run, and audit machine learning experiments
in a research-grade setting. We move from a minimal training loop to a structured,
reproducible experiment pipeline.

The focus is on:

- reproducibility,
- controlled comparisons,
- structured logging,
- and experimental rigor.

---

## Syllabus

1) 🌍 **Big Picture**  
   Why ML experiments are hard today: scale, brittleness, infrastructure

2) 💻 **Dev Setup in 2026**  
   A minimal researcher stack: IDE + AI Assist, Git, environments, tracking

3) 🔁 **Training Script (Vanilla)**  
   Build the minimal loop: data → model → loss → optimizer → eval

4) 📊 **Training Script (Research-Grade)**  
   Make runs comparable: configs, logging, checkpoints, run grids, basic HPO

5) 🧾 *(Optional)* **Working with Text**  
   Run a tiny Transformer experiment (tokenization, batching, evaluation)

6) ⚡ *(Optional)* **Hardware for ML**  
   Scope experiments: VRAM/RAM/disk, throughput bottlenecks, GPU selection

---

## Course Materials

### Slides

[mle4r-winter26.pdf](slides/mle4r-winter26.pdf)

### Notebooks

| # | Section          | Notebook                                                           |
|--:|------------------|--------------------------------------------------------------------|
| 1 | Data             | [1_data.ipynb](notebooks/1_data.ipynb)                             |
| 2 | Model            | [2_model.ipynb](notebooks/2_model.ipynb)                           |
| 3 | Optimizer + Loss | [3_optimizer_and_loss.ipynb](notebooks/3_optimizer_and_loss.ipynb) |
| 4 | Training Loop    | [4_training_loop.ipynb](notebooks/4_training_loop.ipynb)           |
| 5 | Training Script  | [5_training_script.ipynb](notebooks/5_training_script.ipynb)       |
| 6 | Transformers     | [6_transformers.ipynb](notebooks/6_transformers.ipynb)             |

### Scripts

- `train_mnist.py` - single reproducible run
- `runner_simple.py` - programmatic run launcher
- `runner_full.py` - small run grid scheduler
- `hp_opt.py` - minimal random hyperparameter search

These illustrate the progression:

```
One run
↓
Configurable script
↓
Run grid (seed × hyperparameters)
↓
Structured hyperparameter search
```

All materials are self-contained and runnable locally (CPU or single GPU).



---

## Location and Timetable

**📍 Location**  
IMT School for Advanced Studies Lucca  
San Francesco Complex  
Classroom 2

**🗓 Timetable**

| Day    | Date              | Time        |
|--------|-------------------|-------------|
| Friday | February 13, 2026 | 09:00–11:00 |
| Monday | February 16, 2026 | 09:00–11:00 |
