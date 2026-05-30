import os
import numpy as np
from einops import rearrange
from scipy.optimize import differential_evolution

# =========================
# CONFIG
# =========================
MODEL_TAG = "manual slow 1"
CONDITION_TAG = "pretrained"
GAMMA_TAG = "0.6_0.3"
AREAS = ["V1", "V2", "V4", "IT"]
SEEDS = range(1, 100)
SUBSETS = range(1, 6)

INPUT_DIR = "."
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

BINSIZE = 0.033
MAX_LAG = 2.0
TIME_VECTOR = np.arange(BINSIZE, MAX_LAG + BINSIZE, BINSIZE)


# =========================
# HELPERS
# =========================
def load_area_data(area: str, subset: int):
    """Load all available seed files for one area and one subset."""
    matrices = []
    missing_files = []

    for seed in SEEDS:
        filename = (
            f"CORnet-RT_{area}_{MODEL_TAG}_sim{seed}_{CONDITION_TAG}_gamma_{GAMMA_TAG}_subset{subset}_r_t.npy"
        )
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            data = np.load(filepath)
            matrices.append(data)
        except FileNotFoundError:
            missing_files.append(filename)

    if not matrices:
        raise FileNotFoundError(
            f"No files found for area={area}, subset={subset}. "
            f"Example missing file: {missing_files[0] if missing_files else 'N/A'}"
        )

    stacked = np.stack(matrices, axis=1)   # shape: t, n_trials, n_units
    stacked = rearrange(stacked, "t n u -> u n t")  # shape: n_units, n_trials, n_bins
    return stacked, missing_files


def correlation_matrix(unit_activity):
    """unit_activity: n_trials x n_bins"""
    return np.corrcoef(unit_activity, rowvar=False)


def acf(population_activity, max_lag=2.0, binsize=0.033):
    """
    population_activity: n_units x n_trials x n_bins
    returns: mean ACF across units
    """
    max_lag_bins = int(np.round(max_lag / binsize))
    n_units, _, n_bins = population_activity.shape

    correlation_matrix_units = np.zeros((n_units, n_bins, n_bins))
    for u, unit_activity in enumerate(population_activity):
        correlation_matrix_units[u, :, :] = correlation_matrix(unit_activity)

    acf_units = np.zeros((n_units, max_lag_bins))
    for tau in range(max_lag_bins):
        diagonals = np.array(
            [correlation_matrix_units[:, t, t + tau] for t in range(n_bins - tau)]
        )
        acf_units[:, tau] = np.nanmean(diagonals, axis=0)

    return np.nanmean(acf_units, axis=0)


def exponential_cos_model(x, a, b, tau, omega, phi):
    return a * np.exp(-x / tau) * np.cos(2 * np.pi * omega * x + phi) + b


def loss(params, x, y):
    a, b, tau, omega, phi = params
    y_pred = exponential_cos_model(x, a, b, tau, omega, phi)
    return 0.5 * np.linalg.norm(y_pred - y) ** 2


def fit_tau(acf_values, time_vector):
    bounds = [
        (0, 1),             # a
        (-1, 1),            # b
        (0.01, 20),         # tau
        (0, 2),             # omega
        (0, 2 * np.pi),     # phi
    ]

    result = differential_evolution(
        loss,
        bounds,
        args=(time_vector, acf_values),
        strategy="best1bin",
        maxiter=500,
        popsize=250,
        tol=1e-6,
    )
    return result.x  # [a, b, tau, omega, phi]


# =========================
# MAIN LOOP OVER SUBSETS
# =========================
all_acfs = {area: [] for area in AREAS}

for subset in SUBSETS:
    print(f"\nProcessing subset {subset}...")

    autocorrelations = {}
    missing_by_area = {}

    for area in AREAS:
        population_activity, missing_files = load_area_data(area, subset)
        missing_by_area[area] = missing_files
        autocorrelations[area] = acf(
            population_activity,
            max_lag=MAX_LAG,
            binsize=BINSIZE
        )
        all_acfs[area].append(autocorrelations[area])

    best_params = {}
    for area in AREAS:
        best_params[area] = fit_tau(autocorrelations[area], TIME_VECTOR)

    tau_values = {
        f"tau_{area}": best_params[area][2]
        for area in AREAS
    }

    output_file = os.path.join(
        OUTPUT_DIR,
        f"adap_{CONDITION_TAG}_{MODEL_TAG}_subset{subset}_intrinsic_r_t_gamma_{GAMMA_TAG}.txt"
    )

    with open(output_file, "w") as f:
        for key, value in tau_values.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved: {output_file}")
    print(
        "taus:",
        ", ".join([f"{area}={tau_values[f'tau_{area}']:.6f}" for area in AREAS])
    )


acf_arrays = {area: np.stack(all_acfs[area], axis=0) for area in AREAS}

acf_output_file = os.path.join(
    OUTPUT_DIR,
    f"acf_curves_{CONDITION_TAG}_{MODEL_TAG}_gamma_{GAMMA_TAG}.npz"
)
np.savez(
    acf_output_file,
    time=TIME_VECTOR,
    V1=acf_arrays["V1"],
    V2=acf_arrays["V2"],
    V4=acf_arrays["V4"],
    IT=acf_arrays["IT"],
)

print(f"\nSaved all ACF curves: {acf_output_file}")
print(f"Each area array has shape: {acf_arrays['V1'].shape} = (n_subsets, n_bins)")
