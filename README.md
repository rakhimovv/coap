# COAP: Compositional Articulated Occupancy of People

[**Paper**](https://arxiv.org/abs/2204.06184) | [**Video**](https://www.youtube.com/watch?v=qU0q5h6IldU) | [**Project Page**](https://neuralbodies.github.io/COAP)

This is the **UN**official implementation of the CVPR 2022 paper [**COAP: Learning Compositional Occupancy of People**](https://neuralbodies.github.io/COAP)

This repo is mostly my personal playground to play with SMPL models. Please use [**official implementation**](https://github.com/markomih/COAP).
This repose is heavily based on [**LEAP**](https://raw.githubusercontent.com/neuralbodies/leap) code.

# Prerequests 
## 1) SMPL body model
Download a SMPL body model ([**SMPL**](https://smpl.is.tue.mpg.de/), [**SMPL+H**](https://mano.is.tue.mpg.de/), [**SMPL+X**](https://smpl-x.is.tue.mpg.de/), [**MANO**](https://mano.is.tue.mpg.de/)) and store it under `${BODY_MODELS}` directory of the following structure:  
```bash
${BODY_MODELS}
├── smpl
│   └── x
├── smplh
│   ├── male
|   │   └── model.npz
│   ├── female
|   │   └── model.npz
│   └── neutral
|       └── model.npz
├── mano
|   └── x
└── smplx
    └── x
```

NOTE: currently only SMPL+H model is supported. Other models will be available soon.  
 
## Installation

```bash
# note: install the build-essentials package if not already installed (`sudo apt install build-essential`) 
python setup.py build_ext --inplace
pip install -e .
```

## Usage
Follow instructions specified in `data_preparation/README.md` on how to prepare training data.
Then, replace placeholders for pre-defined path variables in configuration files (`configs/*.yml`).

Train: 
```bash
python train_net.py
```

Eval:
```bash
python train_net.py eval_only=true
```