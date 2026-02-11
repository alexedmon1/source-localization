#!/usr/bin/env python3
"""Investigate why sphere ROI has poor conditioning."""

import numpy as np
import pickle
from pathlib import Path

base_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/validation/results/original")

configs_to_compare = [
    ("V21_sphere_roi_sloreta", "sphere", "roi"),
    ("V24_ellipsoid_roi_sloreta", "ellipsoid", "roi"),
    ("V03_sphere_vol_sloreta", "sphere", "vol"),
    ("V10_ellipsoid_vol_sloreta", "ellipsoid", "vol"),
]

print("=" * 100)
print("INVESTIGATING ROI CONDITIONING DIFFERENCES")
print("=" * 100)

for config, bem, source in configs_to_compare:
    print(f"\n{'='*80}")
    print(f"{config}")
    print(f"{'='*80}")

    fwd_path = base_dir / config / "data" / "step4_forward.pkl"
    coords_path = base_dir / config / "data" / "step3_source_coords_mm.npy"

    with open(fwd_path, 'rb') as f:
        fwd = pickle.load(f)

    G = fwd['sol']['data']
    coords = np.load(coords_path)

    n_channels, n_dipoles = G.shape
    n_sources = n_dipoles // 3

    # SVD analysis
    u, s, vh = np.linalg.svd(G, full_matrices=False)

    # Leadfield per source (combining 3 orientations)
    source_leadfield = np.array([np.linalg.norm(G[:, i*3:(i+1)*3]) for i in range(n_sources)])

    # Source depths
    brain_radius = 6.4
    source_distances = np.linalg.norm(coords, axis=1)
    source_depths = brain_radius - source_distances

    print(f"  Sources: {n_sources}")
    print(f"  Condition number: {s.max()/s.min():.2e}")
    print(f"  Singular values: max={s.max():.2e}, min={s.min():.2e}")
    print(f"  Leadfield range: {source_leadfield.max()/source_leadfield.min():.2e}")
    print(f"  Leadfield: min={source_leadfield.min():.2e}, max={source_leadfield.max():.2e}")
    print(f"  Source depths: {source_depths.min():.2f} to {source_depths.max():.2f} mm")

    # Check for very small leadfields (problematic sources)
    threshold = source_leadfield.max() * 0.001  # 0.1% of max
    weak_sources = np.where(source_leadfield < threshold)[0]
    print(f"  Sources with leadfield < 0.1% of max: {len(weak_sources)}")
    if len(weak_sources) > 0:
        print(f"    Weak source depths: {source_depths[weak_sources]}")
        print(f"    Weak source positions (first 5):")
        for idx in weak_sources[:5]:
            print(f"      src {idx}: pos={coords[idx]}, depth={source_depths[idx]:.2f}mm, leadfield={source_leadfield[idx]:.2e}")

    # Correlation with depth
    from scipy.stats import spearmanr
    corr, p = spearmanr(source_depths, source_leadfield)
    print(f"  Depth-leadfield correlation: r={corr:.3f}, p={p:.3e}")

    # Check singular vector patterns
    print(f"\n  Singular vector analysis:")
    print(f"    First singular value explains: {s[0]**2 / np.sum(s**2) * 100:.1f}% of variance")
    print(f"    Top 5 singular values explain: {np.sum(s[:5]**2) / np.sum(s**2) * 100:.1f}% of variance")
    print(f"    Top 10 singular values explain: {np.sum(s[:10]**2) / np.sum(s**2) * 100:.1f}% of variance")

    # Number of significant singular values
    threshold_sv = s.max() * 1e-6
    n_significant = np.sum(s > threshold_sv)
    print(f"    Significant singular values (>1e-6 of max): {n_significant}")

print("\n" + "=" * 100)
print("KEY INSIGHT")
print("=" * 100)
print("""
The condition number reflects how many independent spatial patterns the forward model
can distinguish. A high condition number means some source configurations produce
very similar sensor patterns (hard to distinguish).

For ROI-based sources:
- Sources are clustered within each ROI
- Nearby sources within an ROI can have nearly identical leadfields
- This creates near-singular directions in G, increasing condition number

The ellipsoid BEM has better conditioning because:
- More realistic geometry spreads the leadfield patterns more evenly
- The numerical BEM computation may produce more stable leadfields
""")
