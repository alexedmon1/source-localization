# Regularization Scaling Issue in Mouse EEG Source Localization

**Date**: 2026-01-07
**Status**: Issue identified, fix implemented

## Summary

The default regularization parameters in MNE-Python's inverse methods are calibrated for human-scale EEG data. When applied to mouse brain data, the leadfield values are ~10^10 times larger due to the smaller source-electrode distances, causing the regularization term to become negligible. This results in unreliable inverse solutions for MNE and dSPM methods, while sLORETA remains robust due to its mathematically scale-invariant normalization.

## Physics Background

### Leadfield Scaling

The leadfield matrix G relates source activity to measured EEG:
```
V = G @ J
```

For a current dipole in a spherical conductor, the leadfield scales as:
```
G ~ 1 / (4 * pi * sigma * r^2)
```

where r is the source-electrode distance.

### Mouse vs Human Scale

| Parameter | Human | Mouse | Ratio |
|-----------|-------|-------|-------|
| Brain radius | ~80 mm | ~6.4 mm | 0.08× |
| Source-electrode distance | ~70-100 mm | ~3-8 mm | ~0.05-0.1× |
| Leadfield magnitude | ~10^-7 V/(A·m) | ~10^3 V/(A·m) | ~10^10× |

The ~10^10× larger leadfield in mice is physically correct - smaller distances mean stronger measured signals per unit dipole moment.

## The Problem

### MNE Inverse Solution

The minimum norm estimate uses:
```
W = G.T @ (G @ G.T + lambda^2 * C)^-1
```

where:
- `G` is the leadfield matrix
- `lambda^2` is the regularization parameter (default: SNR=3 → lambda^2 = 1/9 ≈ 0.111)
- `C` is the noise covariance (typically identity or estimated)

### Signal vs Regularization Ratio

For mouse brain data:
- `G @ G.T` eigenvalue (max): ~4.12 × 10^10
- `lambda^2 * C` (typical): ~0.111
- **Ratio**: 370 billion : 1

**The regularization term is effectively ZERO**, meaning:
```
W ≈ G.T @ (G @ G.T)^-1  (pseudoinverse, no regularization)
```

This causes:
1. Numerical instability
2. Noise amplification
3. Loss of depth sensitivity (bias toward superficial sources)

## Method-Specific Behavior

### sLORETA (Scale-Invariant - ROBUST)

sLORETA normalizes source estimates by the resolution matrix diagonal:
```python
resolution_diagonal = diag(W @ G)
source_norm = source / sqrt(resolution_diagonal)
```

When regularization is negligible, `W @ G ≈ G.T @ (G @ G.T)^-1 @ G`, which is a **projection matrix**. Projection matrices have the property that their diagonal elements are bounded (0 to 1), regardless of the scale of G.

**Mathematical proof**:
```
Let P = G.T @ (G @ G.T)^-1 @ G  (projection onto row space of G)
P @ P = P  (idempotent)
trace(P) = rank(G)  (bounded)
0 <= diag(P)_i <= 1  (for all i)
```

Therefore sLORETA normalization is **independent of leadfield magnitude**.

### dSPM (Scale-Dependent - UNRELIABLE)

dSPM normalizes by the noise sensitivity:
```python
noise_norm = sqrt(diag(W @ W.T))
source_norm = source / noise_norm
```

When regularization is negligible:
```
W ≈ G.T @ (G @ G.T)^-1
W @ W.T ≈ G.T @ (G @ G.T)^-2 @ G
```

If we scale G → α*G:
```
W → α^-1 * W  (inverse scaling)
W @ W.T → α^-2 * (W @ W.T)  (quadratic inverse scaling)
```

**Empirical verification**:
```
Scale      dSPM norm mean       sLORETA norm mean
--------------------------------------------------
1          1.272657e-01         5.597076e-01
1000       1.283140e-04         5.611814e-01
```

dSPM normalization drops by 1000× while sLORETA remains constant.

### MNE (Scale-Dependent - UNRELIABLE)

Raw MNE without normalization also suffers from the regularization issue, producing unstable estimates dominated by noise.

## Solution

### Automatic Regularization Scaling (Already Implemented)

Our custom inverse implementations already include auto-scaling regularization:
```python
# From steps/inverse_solution.py and inverse/methods.py
GGT = G @ G.T
GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
```

This scales the identity regularization term by `trace(GGT) / n_channels` (average eigenvalue of GGT), ensuring the regularization is proportional to signal energy regardless of leadfield scale.

### Bug Fix: dSPM Normalization (Fixed 2026-01-07)

The custom dSPM implementation had an incorrect normalization formula:

**WRONG** (was using resolution matrix diagonal):
```python
resolution_diagonal = np.sum(W * G.T, axis=1)  # diag(W @ G)
source_activity_norm = source_activity / resolution_diagonal  # WRONG
```

**CORRECT** (noise sensitivity):
```python
noise_norm = np.sqrt(np.sum(W ** 2, axis=1))  # sqrt(diag(W @ W.T))
source_activity_norm = source_activity / noise_norm  # CORRECT
```

This bug caused dSPM to behave incorrectly regardless of regularization. The fix is in `src/source_localization/steps/inverse_solution.py`.

### Implementation Summary

Both `inverse/methods.py` and `steps/inverse_solution.py` now have:

1. **Auto-scaling regularization**: `lambda2 * trace(GGT) / n_channels * I`
2. **Correct dSPM normalization**: `sqrt(diag(W @ W.T))` (noise sensitivity)
3. **Correct sLORETA normalization**: `sqrt(diag(W @ G))` (resolution matrix diagonal)

## Validation Results

### Before Fix (with dSPM normalization bug)

| Method | ROI Accuracy | Notes |
|--------|-------------|-------|
| sLORETA | 58.1% | Correct normalization, scale-invariant |
| dSPM | 14.2% | **Buggy normalization** (using resolution matrix without sqrt) |
| MNE | 12.8% | No normalization, affected by scale |

### Root Causes Identified

1. **dSPM Bug**: The custom implementation was using `diag(W @ G)` (resolution matrix) instead of `sqrt(diag(W @ W.T))` (noise sensitivity). This completely broke dSPM regardless of regularization.

2. **MNE Scale Sensitivity**: Raw MNE without normalization is inherently sensitive to leadfield scale. The auto-scaling regularization helps but doesn't fully compensate.

3. **sLORETA Robustness**: sLORETA works correctly because its `sqrt(diag(W @ G))` normalization approaches a projection matrix with bounded diagonal elements (0-1), independent of scale.

### After Fix (Quick validation, 2026-01-07)

| Method | ROI Accuracy | Mean Error | Change |
|--------|-------------|------------|--------|
| sLORETA (V03) | **48.0%** | 2.67 mm | (reference) |
| dSPM (V01) | 26.8% | 4.59 mm | +87% vs before |
| MNE (V02) | 22.0% | 4.77 mm | +72% vs before |

**Key findings after fix:**
1. dSPM accuracy nearly doubled (14.2% → 26.8%)
2. sLORETA remains the best method (48.0% vs 26.8% for dSPM)
3. The performance gap is due to sLORETA's inherently scale-invariant normalization

## Recommendations

1. **Use sLORETA** for mouse EEG source localization - it is inherently robust to leadfield scaling
2. **Apply auto-scaling** if using MNE or dSPM methods
3. **Verify regularization ratio** when adapting pipelines to new species/geometries
4. **Check leadfield magnitude** - values ~10^3 V/(A·m) are physically correct for mouse scale

## References

- Pascual-Marqui, R.D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA). Methods Find Exp Clin Pharmacol, 24 Suppl D, 5-12.
- Dale, A.M. et al. (2000). Dynamic statistical parametric mapping. Neuron, 26(1), 55-67.
- Hämäläinen, M.S. & Ilmoniemi, R.J. (1994). Interpreting magnetic fields of the brain. Med Biol Eng Comput, 32(1), 35-42.
