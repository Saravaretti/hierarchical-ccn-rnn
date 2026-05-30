# CORnet-RT with adaptation and an intrinsic timescale hierarchy

Code accompanying "Noisy models of the ventral stream reveal the impact
of recurrence and learned representations on
information processing timescales" (S. Varetti, S. Goldt, E.Piasini, 2026) 

A recurrent model of the ventral stream, with a per-area intrinsic timescale and firing-rate adaptation, reproduces both the intrinsic timescale hierarchy and the hierarchy of stimulus-driven response timescales observed along the visual cortical hierarchy by Piasini et al. (2021), when probed with the same dynamic visual stimuli.


This repository contains:
- a model package (cornet/): the model definition lives in cornet/cornet_rt.py, a 4-area recurrent convolutional network (V1, V2, V4, IT) with a per-area intrinsic timescale, firing-rate adaptation, and intrinsic noise;
- the simulation / feature-extraction script (run.py), at the repository root: it drives the pretrained network with a video stimulus and exports, per area, the time series analysed in the paper.

# Repository structure 

.
├── cornet/             # model package
│   ├── __init__.py     # exposes cornet_rt (and any other CORnet variants you keep)
│   └── cornet_rt.py    # model definition (CORblock_RT, CORnet_RT)
├── run.py              # stimulus loading, simulation, feature extraction
├── data/               # input stimuli (see "Data")
├── requirements.txt
├── LICENSE
└── README.md
