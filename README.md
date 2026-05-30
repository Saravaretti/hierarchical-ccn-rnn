# CORnet-RT with adaptation and leak

Code accompanying "Noisy models of the ventral stream reveal the impact
of recurrence and learned representations on
information processing timescales" (S. Varetti, S. Goldt, E.Piasini, 2026) 

A recurrent model of the ventral stream, with a per-area leak and firing-rate adaptation, reproduces both the intrinsic timescale hierarchy and the hierarchy of stimulus-driven response timescales observed along the rat visual ventral stream in Ref, when probed with the same dynamic visual stimuli.


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
- layer selects the area to record (V1, V2, V4, IT).
- seed sets the simulation index used in the output filename.
- The 5 neuron subsets are drawn with fixed seeds (5–9) inside run.py, so the
subset selection is reproducible across runs.


## Computing response timescales

`acf_response_subsets.py` takes the rate time series produced by `run.py` and
extracts, per area, the response timescale `τ`.

**Input.** The per-unit rate files (`*_r_t.npy`) for areas V1, V2, V4, IT,
across simulation seeds (trials) and neuron subsets. Files are matched by name:

```
CORnet-RT_{area}_{model_tag}_sim{seed}_{condition}_gamma_{gamma_tag}_subset{subset}_r_t.npy
```

For each (area, subset) the script stacks all seeds into a
`(units × trials × bins)` array.

**What it does.** It averages the activity across trials, computes the
autocorrelation function (ACF) of each unit over time lags (up to `MAX_LAG`,
bin size `BINSIZE`), averages the ACF across units, and fits a damped-oscillation
model

```
ACF(t) = a · exp(−t / τ) · cos(2π ω t + φ) + b
```

by differential evolution. The timescale `τ` is the decay constant of this fit.

**Output.** For every subset, a text file with the four per-area timescales:

```
adap_{condition}_{model_tag}_subset{subset}_response_r_t_gamma_{gamma_tag}.txt
   tau_V1: ...
   tau_V2: ...
   tau_V4: ...
   tau_IT: ...
```

and a single archive with the ACF curves used for the fits:

```
acf_curves_response_{condition}_{model_tag}_gamma_{gamma_tag}.npz
   time : lag axis
   V1, V2, V4, IT : arrays of shape (n_subsets, n_bins)
```


## Computing intrinsic timescales

`acf_intrinsic_subset.py` is the intrinsic-timescale counterpart of the
response-timescale script. It shares the same input and fitting procedure; the
only difference is that the autocorrelation is computed on the single-trial
activity, **without** averaging across trials first.

**Input.** The same per-unit rate files (`*_r_t.npy`) for areas V1, V2, V4, IT,
matched by name:

```
CORnet-RT_{area}_{model_tag}_sim{seed}_{condition}_gamma_{gamma_tag}_subset{subset}_r_t.npy
```

For each (area, subset) the script stacks all seeds into a
`(units × trials × bins)` array.

**What it does.** It computes the per-unit autocorrelation function (ACF) over
time lags (up to `MAX_LAG`, bin size `BINSIZE`) directly on the single-trial
data, averages the ACF across units, and fits

```
ACF(t) = a · exp(−t / τ) · cos(2π ω t + φ) + b
```

by differential evolution. The decay constant `τ` is the **intrinsic** timescale
(no trial-averaging is applied, so it reflects the ongoing, noise-driven
correlation structure rather than the stimulus-locked response).

**Output.** For every subset, a text file with the four per-area intrinsic
timescales:

```
adap_{condition}_{model_tag}_subset{subset}_intrinsic_r_t_gamma_{gamma_tag}.txt
   tau_V1: ...
   tau_V2: ...
   tau_V4: ...
   tau_IT: ...
```

and a single archive with the ACF curves used for the fits:

```
acf_curves_{condition}_{model_tag}_gamma_{gamma_tag}.npz
   time : lag axis
   V1, V2, V4, IT : arrays of shape (n_subsets, n_bins)
```


@article{piasini2021temporal,
  title   = {Temporal stability of stimulus representation increases along rodent visual cortical hierarchies},
  author  = {Piasini, Eugenio and Soltuzu, Liviu and Muratore, Paolo and Caramellino, Riccardo and Vinken, Kasper and Op de Beeck, Hans and Balasubramanian, Vijay and Zoccolan, Davide},
  journal = {Nature Communications},
  volume  = {12},
  pages   = {4448},
  year    = {2021},
  doi     = {10.1038/s41467-021-24456-3}
}

