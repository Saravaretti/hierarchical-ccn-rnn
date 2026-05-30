# CORnet-RT with adaptation and an intrinsic timescale hierarchy

Code accompanying "Noisy models of the ventral stream reveal the impact
of recurrence and learned representations on
information processing timescales" (S. Varetti, S. Goldt, E.Piasini, 2026) 

A recurrent model of the ventral stream, with a per-area intrinsic timescale and firing-rate adaptation, reproduces both the intrinsic timescale hierarchy and the hierarchy of stimulus-driven response timescales observed along the visual cortical hierarchy by Piasini et al. (2021), when probed with the same dynamic visual stimuli.


This repository contains:
- a model package (cornet/): the model definition lives in cornet/cornet_rt.py, a 4-area recurrent convolutional network (V1, V2, V4, IT) with a per-area intrinsic timescale, firing-rate adaptation, and intrinsic noise;
- the simulation / feature-extraction script (run.py), at the repository root: it drives the pretrained network with a video stimulus and exports, per area, the time series analysed in the paper.

# Repository structure 

```
.
├── cornet/             # model package
│   ├── __init__.py     # exposes cornet_rt (and any other CORnet variants you keep)
│   └── cornet_rt.py    # model definition (CORblock_RT, CORnet_RT)
├── run.py              # stimulus loading, simulation, feature extraction
├── LICENSE
└── README.md
```

# Requirements

- Python ≥ 3.9
- PyTorch and torchvision
- numpy, h5py, fire, Pillow


# Model 

Each area is a CORblock_RT whose state evolves in discrete time as:
```
adaptation:       s_t = α · s_{t-1} + (1 - α) · r_{t-1}
adapted input:    u_t = conv_input(input) − β · s_t
recurrent state:  h_t = conv1(u_t) + (1 - γ) · h_{t-1} + ξ_t,   ξ_t ~ noise_scale · N(0, 1)
rate:             r_t = ReLU(h_t)
```


Per-area parameters:
| Parameter     | Meaning                              | 
|---------------|--------------------------------------|
| `γ`           | recurrent leak                       |
| `β`           | adaptation strength                  | 
| `α`           | adaptation timescale                 | 
| `noise_scale` | intrinsic noise amplitude            | 




## Data

The main stimulus set comprises nine video clips, each lasting 20 seconds and
sampled at 30 frames per second (fps). The stimuli are described in detail in
[«ref»]. Briefly, six of the videos are naturalistic movies depicting
real-world dynamic scenes, while the remaining three are synthetic controls:
phase-scrambled versions of two natural movies and a white-noise movie.

The stimuli are available at <https://osf.io/7gteq/files/osfstorage>.

`run.py` reads a single HDF5 file containing an `im_matrix` dataset of shape
`(H, W, T)`. Place it in `data/` and pass `--data_path data`.

> Note: the stimulus filename is currently hardcoded in `run.py`. Either rename
> your file accordingly or replace the hardcoded name with a command-line
> argument.

# Usage 

```bash
python run.py --model RT --layer IT --data_path data --output_path outputs --seed 1 --times 601
```
--layer selects the area to record (V1, V2, V4, IT).
--seed sets the simulation index used in the output filename.
The 5 neuron subsets are drawn with fixed seeds (5–9) inside run.py, so the
subset selection is reproducible across runs.


