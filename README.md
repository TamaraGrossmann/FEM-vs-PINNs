# Can Physics-Informed Neural Networks beat the Finite Element Method?

This repository is the official implementation of Can Physics-Informed Neural Networks beat the Finite Element Method? by Tamara G. Grossmann, Urszula Julia Komorowska, Jonas Latz and Carola-Bibiane Schönlieb.

Partial differential equations play a fundamental role in the mathematical modelling of many processes and systems in physical, biological and other sciences. To simulate such processes and systems, the solutions of PDEs often need to be approximated numerically. The finite element method, for instance, is a usual standard methodology to do so. The recent success of deep neural networks at various approximation tasks has motivated their use in the numerical solution of PDEs. These so-called physics-informed neural networks and their variants have shown to be able to successfully approximate a large range of partial differential equations. So far, physics-informed neural networks and the finite element method have mainly been studied in isolation of each other. In this work, we compare the methodologies in a systematic computational study. Indeed, we employ both methods to numerically solve various linear and nonlinear partial differential equations: Poisson in 1D, 2D, and 3D, Allen--Cahn in 1D, semilinear Schrödinger in 1D and 2D.  We then compare computational costs and approximation accuracies. In terms of solution time and accuracy, physics-informed neural networks have not been able to outperform the finite element method in our study. In some experiments, they were faster at evaluating the solved PDE.

If you use this code, please cite:
```bibtex
@article{grossmann2023,
    title={Can Physics-Informed Neural Networks beat the Finite Element Method?},
    author={Tamara G. Grossmann and Urszula Julia Komorowska and Jonas Latz and Carola-Bibiane Schönlieb},
    year={2023}
  }
```
