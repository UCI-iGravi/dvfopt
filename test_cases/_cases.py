"""Static test case definitions — no heavy imports."""

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic test cases (correspondences)
# ---------------------------------------------------------------------------
SYNTHETIC_CASES = {
    # ---- Case 1: Small grids with correspondences ----
    "01a_10x10_crossing": {
        "title": "Case 1a \u2014 10\u00d710 crossing points",
        "msample": np.array([[0, 1, 2], [0, 1, 6]]),
        "fsample": np.array([[0, 3, 6], [0, 3, 2]]),
        "resolution": (10, 10),
    },
    "01b_10x10_opposite": {
        "title": "Case 1b \u2014 10\u00d710 opposite",
        "msample": np.array([[0, 1, 0], [0, 2, 3]]),
        "fsample": np.array([[0, 1, 3], [0, 2, 0]]),
        "resolution": (10, 10),
    },
    "01c_20x40_edges": {
        "title": "Case 1c \u2014 20\u00d740 edges",
        "msample": np.array([
            [0, 2, 10], [0, 3, 5], [0, 6, 4], [0, 11, 5], [0, 15, 7],
            [0, 19, 12], [0, 15, 15], [0, 13, 22], [0, 19, 22], [0, 19, 27],
            [0, 15, 30], [0, 12, 33], [0, 10, 38], [0, 4, 37], [0, 6, 30],
            [0, 1, 33], [0, 5, 26], [0, 2, 23], [0, 1, 19], [0, 2, 14],
        ]),
        "fsample": np.array([
            [0, 1, 7], [0, 5, 11], [0, 6, 1], [0, 15, 1], [0, 12, 2],
            [0, 16, 14], [0, 18, 17], [0, 15, 20], [0, 16, 24], [0, 19, 27],
            [0, 18, 34], [0, 17, 39], [0, 10, 35], [0, 6, 35], [0, 2, 34],
            [0, 3, 30], [0, 2, 26], [0, 2, 23], [0, 3, 20], [0, 2, 14],
        ]),
        "resolution": (20, 40),
    },
    "01d_20x40_crossing": {
        "title": "Case 1d \u2014 20\u00d740 crossing points",
        "msample": np.array([[0, 5, 10], [0, 5, 30]]),
        "fsample": np.array([[0, 15, 30], [0, 15, 10]]),
        "resolution": (20, 40),
    },

    # ---- Case 3: More small grids ----
    "03a_10x10_opposite": {
        "title": "Case 3a \u2014 10\u00d710 opposites",
        "msample": np.array([
            [0, 1, 0], [0, 2, 4],
            [0, 4, 4], [0, 6, 4], [0, 8, 4],
        ]),
        "fsample": np.array([
            [0, 1, 4], [0, 2, 0],
            [0, 4, 0], [0, 6, 0], [0, 8, 0],
        ]),
        "resolution": (10, 10),
    },
    "03b_10x10_crossing": {
        "title": "Case 3b \u2014 10\u00d710 crossing points",
        "msample": np.array([
            [0, 0, 0], [0, 0, 3], [0, 0, 6], [0, 0, 9],
            [0, 6, 0], [0, 6, 3], [0, 6, 6], [0, 6, 9],
        ]),
        "fsample": np.array([
            [0, 2, 3], [0, 2, 0], [0, 2, 9], [0, 2, 6],
            [0, 8, 3], [0, 8, 0], [0, 8, 9], [0, 8, 6],
        ]),
        "resolution": (10, 10),
    },
    "03c_20x20_opposite": {
        "title": "Case 3c \u2014 20\u00d720 opposites",
        "msample": np.array([
            [0, 1, 0], [0, 2, 4],
            [0, 8, 4], [0, 10, 4],
            [0, 16, 4], [0, 17, 0],
            [0, 1, 15], [0, 2, 19],
            [0, 16, 15], [0, 17, 19],
        ]),
        "fsample": np.array([
            [0, 1, 4], [0, 2, 0],
            [0, 8, 0], [0, 10, 0],
            [0, 16, 0], [0, 17, 4],
            [0, 1, 19], [0, 2, 15],
            [0, 16, 19], [0, 17, 15],
        ]),
        "resolution": (20, 20),
    },
    "03d_20x20_crossing": {
        "title": "Case 3d \u2014 20\u00d720 crossing points",
        "msample": np.array([
            [0, 0, 0], [0, 0, 3], [0, 0, 6], [0, 0, 9],
            [0, 0, 11], [0, 0, 14], [0, 0, 16], [0, 0, 19],
            [0, 6, 0], [0, 6, 3], [0, 6, 6], [0, 6, 9],
            [0, 6, 11], [0, 6, 14], [0, 6, 16], [0, 6, 19],
            [0, 12, 0], [0, 12, 3], [0, 12, 6], [0, 12, 9],
            [0, 12, 11], [0, 12, 14], [0, 12, 16], [0, 12, 19],
        ]),
        "fsample": np.array([
            [0, 2, 3], [0, 2, 0], [0, 2, 9], [0, 2, 6],
            [0, 2, 14], [0, 2, 11], [0, 2, 19], [0, 2, 16],
            [0, 8, 3], [0, 8, 0], [0, 8, 9], [0, 8, 6],
            [0, 8, 14], [0, 8, 11], [0, 8, 19], [0, 8, 16],
            [0, 15, 3], [0, 15, 0], [0, 15, 9], [0, 15, 6],
            [0, 15, 14], [0, 15, 11], [0, 15, 19], [0, 15, 16],
        ]),
        "resolution": (20, 20),
    },
}

# Random DVF test cases: (original_shape, new_size, max_magnitude, seed)
RANDOM_DVF_CASES = {
    "01e_20x20_random_spirals": {
        "title": "Case 1e \u2014 20\u00d720 spirals",
        "original_shape": (3, 1, 5, 5),
        "new_size": (20, 20),
        "max_magnitude": 5.0,
        "seed": 42,
    },
    "01f_20x20_random_seed_42": {
        "title": "Case 1f \u2014 20\u00d720 random seed 42",
        "original_shape": (3, 1, 20, 20),
        "new_size": None,  # no rescaling
        "max_magnitude": 3.0,
        "seed": 42,
    },
    "03a_10x10_random_seed_42": {
        "title": "Case 3a \u2014 10\u00d710 spirals (seed 42)",
        "original_shape": (3, 1, 5, 5),
        "new_size": (10, 10),
        "max_magnitude": 5.0,
        "seed": 42,
    },
    "03c_20x20_random_seed_42": {
        "title": "Case 3c \u2014 20\u00d720 spirals (seed 42)",
        "original_shape": (3, 1, 5, 5),
        "new_size": (20, 20),
        "max_magnitude": 5.0,
        "seed": 42,
    },
}

# Real-data slice definitions: (slice_idx, scale_factor, label_suffix)
REAL_DATA_SLICES = {
    "02a_64x91": {"slice_idx": 90, "scale_factor": 0.2, "title": "Case 2a \u2014 slice 90 (64\u00d791)"},
    "02a_320x456": {"slice_idx": 90, "scale_factor": 1.0, "title": "Case 2a \u2014 slice 90 (320\u00d7456)"},
    "02b_64x91": {"slice_idx": 200, "scale_factor": 0.2, "title": "Case 2b \u2014 slice 200 (64\u00d791)"},
    "02b_320x456": {"slice_idx": 200, "scale_factor": 1.0, "title": "Case 2b \u2014 slice 200 (320\u00d7456)"},
    "02c_64x91": {"slice_idx": 350, "scale_factor": 0.2, "title": "Case 2c \u2014 slice 350 (64\u00d791)"},
    "02c_320x456": {"slice_idx": 350, "scale_factor": 1.0, "title": "Case 2c \u2014 slice 350 (320\u00d7456)"},
    "02d_64x91": {"slice_idx": 500, "scale_factor": 0.2, "title": "Case 2d \u2014 slice 500 (64\u00d791)"},
    "02d_320x456": {"slice_idx": 500, "scale_factor": 1.0, "title": "Case 2d \u2014 slice 500 (320\u00d7456)"},
}
