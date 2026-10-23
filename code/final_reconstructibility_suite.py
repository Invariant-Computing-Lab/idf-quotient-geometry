#!/usr/bin/env python3
"""
Final reconstructibility and finite-resolution suite for the invariant reformulation
of the Intrinsic Dimensionality Framework (IDF).

Scope:
- Synthetic mathematical validation only.
- No empirical Standard Model validation.
- No claim of new-particle discovery or physical correctness.

Experiments:
B1. Fingerprint-collision benchmark.
B2. Leave-one-invariant-out reconstructibility.
B3. Spectral truncation and noise robustness.
B4. Dense torus-modulus recovery.
B5. Global scale-quotient validation of spectral fingerprints.
B6. Critical spectral-gap collapse along torus degeneration.
B7. Finite-resolution collision analysis.

Seed: 20260812
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20260812
rng = np.random.default_rng(SEED)

def sphere_spectrum(k=20):
    vals = []
    ell = 1
    while len(vals) < k:
        vals.extend([ell * (ell + 1)] * (2 * ell + 1))
        ell += 1
    return np.asarray(vals[:k], dtype=float)

def torus_spectrum(aspect=1.0, k=20, modes=16):
    r = float(aspect)
    vals = []
    for m in range(-modes, modes + 1):
        for n in range(-modes, modes + 1):
            if m == 0 and n == 0:
                continue
            vals.append(4 * np.pi**2 * (m*m/r + n*n*r))
    return np.sort(np.asarray(vals, dtype=float))[:k]

def spectral_ratios(vals, k=None):
    vals = np.asarray(vals, dtype=float)
    if k is not None:
        vals = vals[:k]
    return vals / vals[0]

def fingerprint_recovery(df, cols):
    keys = []
    for _, row in df.iterrows():
        key = []
        for c in cols:
            x = row[c]
            if isinstance(x, np.ndarray):
                x = tuple(np.round(x, 10))
            key.append(x)
        keys.append(tuple(key))
    unique = len(set(keys))
    return unique / len(df), unique

def main(outdir="results"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)

    geom_defs = [("sphere", 0, None)] + [
        ("torus", 1, r) for r in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    ]
    gauge_defs = [("U1", 1, 1), ("SU2", 3, 1), ("SU3", 8, 2)]
    degrees = [0, 1, 2]

    rec = []
    for gname, genus, aspect in geom_defs:
        spec = sphere_spectrum(12) if gname == "sphere" else torus_spectrum(aspect, 12)
        ratios = spectral_ratios(spec, 8)
        for gauge, gdim, grank in gauge_defs:
            for degree in degrees:
                rec.append({
                    "geometry": gname if aspect is None else f"{gname}_r{aspect:g}",
                    "genus": genus,
                    "gauge": gauge,
                    "gauge_dim": gdim,
                    "gauge_rank": grank,
                    "degree": degree,
                    "spec_ratios": ratios,
                    "class_id": f"{gname}:{aspect}|{gauge}|d{degree}",
                })
    classes = pd.DataFrame(rec)
    nclasses = len(classes)

    group_defs = {
        "topology": ["genus"],
        "gauge": ["gauge_dim", "gauge_rank"],
        "degree": ["degree"],
        "spectrum": ["spec_ratios"],
    }

    # B1: staged fingerprint collisions
    levels = {
        "F0_topology": ["genus"],
        "F1_topology_gauge": ["genus", "gauge_dim", "gauge_rank"],
        "F2_plus_degree": ["genus", "gauge_dim", "gauge_rank", "degree"],
        "F3_plus_spectrum": ["genus", "gauge_dim", "gauge_rank", "degree", "spec_ratios"],
    }
    collision = []
    for name, cols in levels.items():
        recovery, unique = fingerprint_recovery(classes, cols)
        collision.append({
            "fingerprint": name,
            "unique_fingerprints": unique,
            "n_classes": nclasses,
            "max_exact_recovery": recovery,
            "collision_rate": 1 - recovery,
        })
    collision_df = pd.DataFrame(collision)
    collision_df.to_csv(out / "fingerprint_collision_results.csv", index=False)

    # B2: all subsets + leave-one-invariant-out
    subset_rows = []
    for size in range(1, len(group_defs) + 1):
        for names in itertools.combinations(group_defs, size):
            cols = sum((group_defs[n] for n in names), [])
            recovery, unique = fingerprint_recovery(classes, cols)
            subset_rows.append({
                "groups": "+".join(names),
                "n_groups": size,
                "exact_recovery": recovery,
                "unique": unique,
            })
    subset_df = pd.DataFrame(subset_rows).sort_values(["n_groups", "groups"])
    subset_df.to_csv(out / "fingerprint_subset_results.csv", index=False)

    ablation_rows = []
    for drop in group_defs:
        keep = [x for x in group_defs if x != drop]
        cols = sum((group_defs[n] for n in keep), [])
        recovery, unique = fingerprint_recovery(classes, cols)
        ablation_rows.append({
            "dropped": drop,
            "kept": "+".join(keep),
            "exact_recovery": recovery,
            "unique": unique,
        })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out / "leave_one_invariant_out_results.csv", index=False)

    # Save synthetic class table with explicit spectral columns
    classes_out = classes.copy()
    for i in range(8):
        classes_out[f"spec_r{i+1}"] = [x[i] for x in classes_out["spec_ratios"]]
    classes_out = classes_out.drop(columns=["spec_ratios"])
    classes_out.to_csv(out / "synthetic_identity_classes.csv", index=False)

    # Geometry templates
    templates = []
    for gname, genus, aspect in geom_defs:
        vals = sphere_spectrum(20) if gname == "sphere" else torus_spectrum(aspect, 20)
        templates.append((
            gname if aspect is None else f"{gname}_r{aspect:g}",
            spectral_ratios(vals, 20),
        ))
    template_names = [x[0] for x in templates]
    template_mat = np.vstack([x[1] for x in templates])

    # B3: truncation + noise
    robust = []
    for K in [2, 3, 4, 6, 8, 12]:
        base_feat = np.log(template_mat[:, :K])[:, 1:]
        scale = np.std(base_feat, axis=0, ddof=1)
        scale = np.where(scale < 1e-12, 1.0, scale)
        for noise in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
            correct = 0
            total = 0
            for i in range(len(template_names)):
                for _ in range(500):
                    raw = template_mat[i, :K] * np.exp(rng.normal(0, noise, size=K))
                    noisy = raw / raw[0]
                    feat = np.log(noisy)[1:]
                    dist = np.sum(((base_feat - feat) / scale)**2, axis=1)
                    pred = int(np.argmin(dist))
                    correct += int(pred == i)
                    total += 1
            robust.append({"K": K, "noise_sigma": noise, "accuracy": correct / total})
    robust_df = pd.DataFrame(robust)
    robust_df.to_csv(out / "spectral_noise_truncation_results.csv", index=False)

    # B4: dense torus-modulus recovery
    rgrid = np.exp(np.linspace(np.log(1.0), np.log(4.0), 161))
    dense = np.vstack([spectral_ratios(torus_spectrum(r, 20), 8) for r in rgrid])
    dense_feat = np.log(dense[:, 1:])
    dense_scale = np.std(dense_feat, axis=0, ddof=1)
    dense_scale = np.where(dense_scale < 1e-12, 1.0, dense_scale)
    dense_rows = []
    for noise in [0.001, 0.005, 0.01, 0.02, 0.05]:
        rel_err = []
        exact = 0
        for _ in range(8000):
            idx = int(rng.integers(0, len(rgrid)))
            raw = dense[idx] * np.exp(rng.normal(0, noise, size=8))
            noisy = raw / raw[0]
            feat = np.log(noisy[1:])
            dist = np.sum(((dense_feat - feat) / dense_scale)**2, axis=1)
            pred = int(np.argmin(dist))
            exact += int(pred == idx)
            rel_err.append(abs(rgrid[pred] - rgrid[idx]) / rgrid[idx])
        dense_rows.append({
            "noise_sigma": noise,
            "exact_grid_accuracy": exact / 8000,
            "median_relative_aspect_error": float(np.median(rel_err)),
            "p95_relative_aspect_error": float(np.quantile(rel_err, 0.95)),
        })
    dense_df = pd.DataFrame(dense_rows)
    dense_df.to_csv(out / "dense_moduli_recovery_results.csv", index=False)

    # B5: scale quotient
    base = torus_spectrum(2.0, 20)
    scale_rows = []
    for s in np.exp(rng.uniform(-5, 5, size=5000)):
        eig = base / s**2
        area = s**2
        raw_err = np.max(np.abs(eig - base) / base)
        area_norm_err = np.max(np.abs(area * eig - base) / base)
        ratio_err = np.max(
            np.abs((eig / eig[0]) - (base / base[0])) / (base / base[0])
        )
        scale_rows.append((raw_err, area_norm_err, ratio_err))
    scale_arr = np.asarray(scale_rows)

    # B6: critical spectral-gap collapse
    rcrit = np.logspace(0, 4, 600)
    lam1 = np.array([torus_spectrum(r, 1)[0] for r in rcrit])
    slope, intercept = np.polyfit(np.log(rcrit[50:]), np.log(lam1[50:]), 1)

    # B7: finite-resolution collisions
    quant = []
    for K in [3, 4, 6, 8, 12]:
        specs = np.vstack([spectral_ratios(torus_spectrum(r, 20), K) for r in rgrid])
        for decimals in [1, 2, 3, 4, 6]:
            keys = [tuple(np.round(x, decimals)) for x in specs]
            unique = len(set(keys))
            quant.append({
                "K": K,
                "round_decimals": decimals,
                "unique_of_161": unique,
                "collision_rate": 1 - unique / len(rgrid),
            })
    quant_df = pd.DataFrame(quant)
    quant_df.to_csv(out / "finite_resolution_collision_results.csv", index=False)

    minimal = subset_df[subset_df["exact_recovery"] == 1.0].sort_values(
        ["n_groups", "groups"]
    ).iloc[0]

    summary = {
        "seed": SEED,
        "synthetic_identity_classes": int(nclasses),
        "minimal_full_recovery_subset_in_benchmark": minimal["groups"],
        "minimal_full_recovery_n_groups": int(minimal["n_groups"]),
        "raw_spectrum_scale_max_relative_change": float(scale_arr[:, 0].max()),
        "area_normalized_spectrum_scale_max_relative_error": float(scale_arr[:, 1].max()),
        "ratio_spectrum_scale_max_relative_error": float(scale_arr[:, 2].max()),
        "critical_loglog_slope_lambda1_vs_aspect": float(slope),
        "critical_drop_factor": float(lam1[0] / lam1[-1]),
    }
    pd.DataFrame([summary]).to_csv(out / "final_summary.csv", index=False)
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figures
    f, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(collision_df["fingerprint"], 100 * collision_df["max_exact_recovery"])
    ax.set_ylabel("Maximum exact class recovery (%)")
    ax.set_xlabel("Fingerprint level")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Synthetic fingerprint sufficiency benchmark")
    f.tight_layout()
    f.savefig(figdir / "fig1_fingerprint_recovery.png", dpi=240)
    plt.close(f)

    f, ax = plt.subplots(figsize=(7.2, 4.4))
    for K in [2, 4, 8, 12]:
        d = robust_df[robust_df["K"] == K]
        ax.plot(100 * d["noise_sigma"], 100 * d["accuracy"], marker="o", label=f"K={K}")
    ax.set_xlabel("Multiplicative spectral noise sigma (%)")
    ax.set_ylabel("Geometry-class recovery (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Noise robustness versus spectral truncation")
    ax.legend()
    f.tight_layout()
    f.savefig(figdir / "fig2_noise_robustness.png", dpi=240)
    plt.close(f)

    f, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(rcrit, lam1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rectangular torus aspect ratio r")
    ax.set_ylabel("Area-normalized first eigenvalue A lambda1")
    ax.set_title("Spectral-gap collapse along torus degeneration")
    f.tight_layout()
    f.savefig(figdir / "fig3_critical_locus.png", dpi=240)
    plt.close(f)

    f, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        100 * dense_df["noise_sigma"],
        100 * dense_df["median_relative_aspect_error"],
        marker="o",
        label="Median",
    )
    ax.plot(
        100 * dense_df["noise_sigma"],
        100 * dense_df["p95_relative_aspect_error"],
        marker="o",
        label="95th percentile",
    )
    ax.set_xlabel("Multiplicative spectral noise sigma (%)")
    ax.set_ylabel("Relative aspect-ratio error (%)")
    ax.set_title("Continuous torus-modulus recoverability")
    ax.legend()
    f.tight_layout()
    f.savefig(figdir / "fig4_dense_moduli_recovery.png", dpi=240)
    plt.close(f)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
