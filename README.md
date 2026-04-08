# deep-learning-1
Systematic study of deep learning methodology (regularization &amp; optimization) applied to CIFAR-10 image classification. Covers: MLP from scratch, L1/L2/Dropout/BatchNorm comparison, optimizer shootout (SGD→Adam), and learning rate scheduling.
# CIFAR-10 Deep Learning Study
### SWE012 — Deep Learning with Python
**Istinye University, Department of Computer Engineering**

## Quick Start
See [REPORT.md](REPORT.md) for the full project documentation,
methodology, and results.

## Repository Structure
├── REPORT.md              # Full project report
├── notebook.ipynb         # Google Colab notebook
├── responsibilities/      # Individual contributions
│   ├── 220911692.md
│   ├── 220911742.md
│   ├── 210911197.md
│   └── 210911108.md
└── figures/               # All experiment figures

## Results Summary
| Model | Test Accuracy |
|-------|--------------|
| MLP Baseline | 54.0% |
| Best MLP (BatchNorm) | 59.2% |
| CNN (Adam + Warmup) | **79.5%** |
