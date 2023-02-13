# Can Physics-Informed Neural Networks beat the Finite Element Method?

This repository is the official implementation of Can Physics-Informed Neural Networks beat the Finite Element Method? by Tamara G. Grossmann, Urszula Julia Komorowska, Jonas Latz and Carola-Bibiane Schönlieb.

Partial differential equations play a fundamental role in the mathematical modelling of many processes and systems in physical, biological and other sciences. To simulate such processes and systems, the solutions of PDEs often need to be approximated numerically. The finite element method, for instance, is a usual standard methodology to do so. The recent success of deep neural networks at various approximation tasks has motivated their use in the numerical solution of PDEs. These so-called physics-informed neural networks and their variants have shown to be able to successfully approximate a large range of partial differential equations. So far, physics-informed neural networks and the finite element method have mainly been studied in isolation of each other. In this work, we compare the methodologies in a systematic computational study. Indeed, we employ both methods to numerically solve various linear and nonlinear partial differential equations: Poisson in 1D, 2D, and 3D, Allen--Cahn in 1D, semilinear Schrödinger in 1D and 2D.  We then compare computational costs and approximation accuracies. In terms of solution time and accuracy, physics-informed neural networks have not been able to outperform the finite element method in our study. In some experiments, they were faster at evaluating the solved PDE.

If you use this code, please cite:
```bibtex
@misc{grossmann2023,
  doi = {10.48550/ARXIV.2302.04107},
  url = {https://arxiv.org/abs/2302.04107},
  author = {Grossmann, Tamara G. and Komorowska, Urszula Julia and Latz, Jonas and Schönlieb, Carola-Bibiane},
  keywords = {Numerical Analysis (math.NA), Machine Learning (cs.LG), FOS: Mathematics, FOS: Mathematics, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Can Physics-Informed Neural Networks beat the Finite Element Method?},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Requirements FEM codes

The code was written in Python using the FEniCS toolbox (https://fenicsproject.org/). Following packages are required to run the code smoothly:
- FEniCS based on dolfin version 2019.1.0
- numpy

## Requirements PINN codes

The code was written in Pytorch using the jax library for deep learning. Following packages are required to run the code smoothly:
- jax
- flax
- optax
- tensorflow probability for jax
- numpy
- pyDOE
- scipy

## Evaluation points

Due to file size, we can only upload the evalutation points and solution matrices for Poisson 1D, 2D, Allen-Cahn 1D and Schrödinger 1D. However, the evalutation points and solution matrices for the remaining PDEs can be easily created using FEniCS, as they are based on the meshes generated through the code. Details of mesh size and temporal resolution can be found in the paper. 

