#!/usr/bin/env python3
"""Analyze condition numbers across all validation configurations."""

import numpy as np
import pickle
from pathlib import Path

base_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/validation/results/original")

# All configs
configs = [
    "V01_sphere_vol_dspm",
    "V02_sphere_vol_mne",
    "V03_sphere_vol_sloreta",
    "V04_sphere_surf_dspm",
    "V05_sphere_surf_mne",
    "V06_sphere_surf_sloreta",
    "V07_sphere_roi_dspm",
    "V08_ellipsoid_vol_dspm",
    "V09_ellipsoid_vol_mne",
    "V10_ellipsoid_vol_sloreta",
    "V11_ellipsoid_surf_dspm",
    "V12_ellipsoid_surf_mne",
    "V13_ellipsoid_surf_sloreta",
    "V14_ellipsoid_roi_dspm",
    "V15_ellipsoid_roi_mne",
    "V16_sphere_vol_lcmv",
    "V17_sphere_vol_dics",
    "V18_ellipsoid_vol_lcmv",
    "V19_ellipsoid_vol_dics",
    "V20_sphere_roi_mne",
    "V21_sphere_roi_sloreta",
    "V22_sphere_roi_lcmv",
    "V23_sphere_roi_dics",
    "V24_ellipsoid_roi_sloreta",
    "V25_ellipsoid_roi_lcmv",
    "V26_ellipsoid_roi_dics",
]

print("=" * 100)
print("CONDITION NUMBER ANALYSIS ACROSS ALL CONFIGURATIONS")
print("=" * 100)
print()
print("Note: Condition number is a property of the leadfield matrix G, NOT the inverse method.")
print("      Same BEM + source space = same condition number regardless of inverse method.")
print()

results = []

for config in configs:
    fwd_path = base_dir / config / "data" / "step4_forward.pkl"
    if not fwd_path.exists():
        print(f"Skipping {config}: forward model not found")
        continue

    with open(fwd_path, 'rb') as f:
        fwd = pickle.load(f)

    G = fwd['sol']['data']  # (n_channels, n_dipoles)
    n_channels, n_dipoles = G.shape
    n_sources = n_dipoles // 3

    # Parse config name
    parts = config.split('_')
    bem_type = parts[1]  # sphere or ellipsoid
    source_type = parts[2]  # vol, surf, or roi
    method = parts[3]  # dspm, mne, sloreta, lcmv, dics

    # Compute condition number of G
    # Method 1: SVD-based (most accurate)
    u, s, vh = np.linalg.svd(G, full_matrices=False)
    cond_svd = s.max() / s.min()

    # Method 2: Condition number of G @ G.T (Gram matrix)
    GGT = G @ G.T
    eigenvalues = np.linalg.eigvalsh(GGT)
    cond_GGT = eigenvalues.max() / eigenvalues.min()

    # Leadfield statistics
    G_norms = np.linalg.norm(G, axis=0)  # norm per dipole
    G_range = G_norms.max() / G_norms.min()

    results.append({
        'config': config,
        'bem': bem_type,
        'source': source_type,
        'method': method,
        'n_sources': n_sources,
        'cond_G': cond_svd,
        'cond_GGT': cond_GGT,
        'G_range': G_range,
        's_max': s.max(),
        's_min': s.min(),
    })

# Group by BEM + source type (since method doesn't affect condition number)
print("\n" + "=" * 100)
print("CONDITION NUMBERS BY BEM + SOURCE SPACE")
print("=" * 100)
print()
print(f"{'BEM':<12} {'Source':<12} {'Sources':>8} {'cond(G)':>12} {'cond(G@G.T)':>14} {'G range':>12} {'σ_max':>12} {'σ_min':>10}")
print("-" * 100)

seen = set()
for r in results:
    key = (r['bem'], r['source'])
    if key in seen:
        continue
    seen.add(key)
    print(f"{r['bem']:<12} {r['source']:<12} {r['n_sources']:>8} {r['cond_G']:>12.2e} {r['cond_GGT']:>14.2e} {r['G_range']:>12.2e} {r['s_max']:>12.2e} {r['s_min']:>10.2e}")

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print()

# Condition number interpretation
print("Condition number interpretation:")
print("  - cond(G) < 100: Well-conditioned (excellent)")
print("  - cond(G) 100-1000: Moderate conditioning (good)")
print("  - cond(G) 1000-10000: Poorly conditioned (acceptable with regularization)")
print("  - cond(G) > 10000: Ill-conditioned (problematic)")
print()

# Compare sphere vs ellipsoid
sphere_conds = [r['cond_G'] for r in results if r['bem'] == 'sphere']
ellipsoid_conds = [r['cond_G'] for r in results if r['bem'] == 'ellipsoid']

print(f"Sphere BEM condition numbers: {min(sphere_conds):.2e} to {max(sphere_conds):.2e}")
print(f"Ellipsoid BEM condition numbers: {min(ellipsoid_conds):.2e} to {max(ellipsoid_conds):.2e}")

# Compare source types
for source_type in ['vol', 'surf', 'roi']:
    conds = [r['cond_G'] for r in results if r['source'] == source_type]
    if conds:
        print(f"{source_type.upper()} source space condition numbers: {min(conds):.2e} to {max(conds):.2e}")

# Check what I reported earlier
print("\n" + "=" * 100)
print("VERIFICATION: What is the 4×10^5 condition number from?")
print("=" * 100)
print()
print("The 4×10^5 value was cond(G @ G.T) for sphere volumetric.")
print("This is the SQUARE of cond(G) because eigenvalues of G@G.T are singular values of G squared.")
print()
print("Actual condition numbers of G:")
for r in results:
    if r['source'] == 'vol':
        print(f"  {r['bem']} {r['source']}: cond(G) = {r['cond_G']:.2e}, cond(G@G.T) = {r['cond_GGT']:.2e}")
        break
