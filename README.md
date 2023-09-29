# InverseFWM: Inverse-design for spontaneous Four-Wave Mixing
Inverse-design for spontaneous four-wave mixing based on EMopt. 

Code used for optimizing the device demonstrated in the paper: [Interpretable inverse-designed cavity for on-chip nonlinear and quantum optics](https://arxiv.org/abs/2308.03036)

# Overview
We implement an inverse-design approach to amplify the efficiency of on-chip photon pair generation using the open-source package EmOpt. Our method employs a multi-frequency co-optimization strategy and calculates gradients with respect to the design parameters via the adjoint method. The resulting efficiency enhancement stems not only from the increased field intensity due to the confinement of light in high quality factor cavity resonances but also from the improvement of phase-matching conditions, along with coupling between the cavity and waveguide mode considered in the design.

# Usage
The user needs first to install [EMopt](https://github.com/anstmichaels/emopt).

# Credits
If you use this for your research or work, please cite:

Jia, Z., Qarony, W., Park, J., Hooten, S., Wen, D., Zhiyenbayev, Y., ... & Kanté, B. (2023). Interpretable inverse-designed cavity for on-chip nonlinear and quantum optics. arXiv preprint arXiv:2308.03036.

