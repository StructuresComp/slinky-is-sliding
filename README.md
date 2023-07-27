# slinky-is-sliding
This repository contains the code for the paper:
- [Rapidly encoding generalizable dynamics in a Euclidean symmetric neural network: a Slinky case study](https://doi.org/10.1016/j.eml.2022.101925)

In this work, we propose a physics-informed deep learning approach to build reduced-order models of physical systems. We use Slinky as a demonstration. The approach introduces a **Euclidena symmetric neural network architecture (ESNN)**, trained under the **neural ordinary differential equation** framework. The ESNN implements a physics-guided architecture that simultaneously preserves energy invariance and force equivariance on Euclidean transformations of the input, including translation, rotation, and reflection. We demonstrate that the ESNN approach is able to accelerate simulation by roughly **60** times compared to traditional numerical methods and achieve a superior generalization performance, i.e., the neural network, trained on a single demonstration case, predicts accurately on unseen cases with different Slinky configurations and boundary conditions.

![](./media/40.gif)

## Requirements
To run the code, you must install the following dependencies first:
- PyTorch (1.8.1)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Files
- `main.py` trains the Euclidean symmetric neural network
- `slinky/neuralnets.py` contains the DenseNet-like structure
- `slinky/transformation.py` contains the rigid body motion removal module and chirality module
- `slinky/func.py` contains ODEFunc for generating equivariant surrogate forces using Euclidean symmetric neural networks and ODEPhys for generating derivatives used by ODE solvers
- `slinky/misc.py` contains utility functions
- `PlotHist.m` is the MATLAB code for visualizing training loss history
- `VisualizeData_Slinky.m` is the MATLAB code for visualizing training results (Slinky motion)

## Citation
If you use this code for part of your project or paper, or get inspired by the associated paper, please cite:  

    @article{Li2023Rapidly,
        title = {Rapidly encoding generalizable dynamics in a {E}uclidean symmetric neural network},
        journal = {Extreme Mechanics Letters},
        volume = {58},
        pages = {101925},
        year = {2023},
        issn = {2352-4316},
        doi = {https://doi.org/10.1016/j.eml.2022.101925},
        url = {https://www.sciencedirect.com/science/article/pii/S2352431622002012},
        author = {Qiaofeng Li and Tianyi Wang and Vwani Roychowdhury and M. Khalid Jawed},
        keywords = {Reduced-order model, Data-driven, Deep learning, Neural network, Neural ordinary differential equation},
    }