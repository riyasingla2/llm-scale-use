"""Layer-wise linear probes for designed valence, plus a read-out on the self item.

For each layer, a ridge regression is trained to predict a vignette's
designed level (1..5) from the last-token residual-stream activation. It is
evaluated with leave-one-domain-out cross-validation: train on work, health
and relationships, test on money, and so on. A probe that only memorised
domain-specific wording would fail this.

The best layer's probe is then applied to the self item, which gives an
"internal" reading of the model's self-rating to compare with its verbal
answer. This is in the spirit of the truth-classifier approach of Kaiser &
Enderby (2026), with one big caveat: a probe trained on third-person
vignettes may not transfer to first-person self-reports. Treat the
self read-out as exploratory.

    python -m scaleuse.probe results/wb_Qwen2.5-0.5B-Instruct.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import vignette_adjusted
from .items import SELF_ID

ALPHAS = np.logspace(-1, 5, 13)


def make_probe():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def layerwise_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """X: [n, L, d] activations; y: [n] levels; groups: [n] CV groups.

    Returns out-of-fold Spearman rho and R^2 per layer.
    """
    out = []
    for layer in range(X.shape[1]):
        Xl = X[:, layer, :].astype(np.float64)
        pred = np.empty_like(y, dtype=float)
        for g in np.unique(groups):
            te = groups == g
            m = make_probe().fit(Xl[~te], y[~te])
            pred[te] = m.predict(Xl[te])
        rho = stats.spearmanr(pred, y).statistic
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append(dict(layer=layer, oof_spearman=rho, oof_r2=r2))
    return pd.DataFrame(out)


def transfer_probe(X_rate: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   X_sum: np.ndarray) -> pd.DataFrame:
    """Leave-one-domain-out: train on rate activations, test on summarise activations.

    X_rate and X_sum must share the same vignette ordering and domain groups.
    Returns Spearman rho per layer (no R^2 since this is a transfer, not in-sample CV).
    """
    out = []
    for layer in range(X_rate.shape[1]):
        pred = np.empty_like(y, dtype=float)
        for g in np.unique(groups):
            te = groups == g
            m = make_probe().fit(X_rate[~te, layer, :].astype(np.float64), y[~te])
            pred[te] = m.predict(X_sum[te, layer, :].astype(np.float64))
        rho = stats.spearmanr(pred, y).statistic
        out.append(dict(layer=layer, spearman=rho))
    return pd.DataFrame(out)


def permutation_p(X, y, groups, observed_best, n_perm=500, seed=0):
    """Share of permutations where the MAX layer rho >= the observed best-layer rho.

    Runs layerwise_probe over ALL layers per permutation to account for the
    selection of the best layer from the observed data.
    Returns (p_value, count); count == 0 means p < 1/(n_perm+1).
    """
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        null_rhos = layerwise_probe(X, yp, groups)["oof_spearman"].to_numpy()
        if not np.isnan(null_rhos).all():
            count += int(np.nanmax(null_rhos) >= observed_best)
    return (count + 1) / (n_perm + 1), count


def logit_vignette_adjusted_self(logit_path: Path, fmt: str) -> float | None:
    """Vignette-adjusted logit expected self-rating on the 1..5 level scale.

    Loads the model's logit expected ratings from `logit_path`, builds the
    mean vignette expected_norm per designed level as the scale profile, then
    places the self item's expected_norm on the 1..5 level scale by linear
    interpolation (monotone via np.maximum.accumulate inside vignette_adjusted).
    Returns None if the file doesn't exist or data is insufficient.
    """
    if not Path(logit_path).exists():
        return None
    lr = pd.read_json(logit_path, lines=True)
    lr = lr[lr["fmt"] == fmt]
    vig_lr = lr[lr["item_id"] != SELF_ID]
    self_lr = lr[lr["item_id"] == SELF_ID]
    if self_lr.empty or vig_lr.empty:
        return None
    profile = vig_lr.groupby("level")["expected_norm"].mean().sort_index()
    return vignette_adjusted(float(self_lr["expected_norm"].iloc[0]), profile)


def _load_vignettes(npz_path, fmt):
    """Load vignette activations and labels from an npz, filtered to fmt.

    If the file contains only one unique fmt value (e.g. a summarise npz), all
    rows are used and ``fmt`` is ignored.  If ``fmt`` matches no rows after
    filtering, exits with a message listing available fmt values.
    Returns effective_fmt as the last element.
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    states = z["states"]
    available = sorted(meta["fmt"].unique().tolist())
    effective_fmt = available[0] if len(available) == 1 else fmt
    sel = (meta["fmt"] == effective_fmt).to_numpy()
    if not sel.any():
        sys.exit(f"--fmt '{fmt}' not found in {npz_path}. Available: {available}")
    meta, states = meta[sel].reset_index(drop=True), states[sel]
    vig = (meta["item_id"] != SELF_ID).to_numpy()
    X = states[vig]
    y = meta.loc[vig, "level"].to_numpy(float)
    g = meta.loc[vig, "domain"].to_numpy()
    return X, y, g, states, meta, vig, effective_fmt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--fmt", default="0-10")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--transfer", action="store_true",
                    help="compare rate vs summarise vs cross-task transfer side by side")
    a = ap.parse_args(argv)

    if a.transfer:
        summarise_npz = a.npz.with_name(a.npz.stem + "_summarise.npz")
        X_rate, y, g, _, _, _, _ = _load_vignettes(a.npz, a.fmt)
        zs = np.load(summarise_npz, allow_pickle=False)
        meta_s = pd.DataFrame(json.loads(str(zs["meta"])))
        X_sum = zs["states"].astype(np.float32)
        assert np.array_equal(y, meta_s["level"].to_numpy(float)), \
            "rate and summarise npz have different vignette ordering"

        res_rate = layerwise_probe(X_rate, y, g)
        res_sum = layerwise_probe(X_sum, y, meta_s["domain"].to_numpy())
        res_xfer = transfer_probe(X_rate, y, g, X_sum)

        combined = pd.DataFrame({
            "layer":     res_rate["layer"],
            "rate":      res_rate["oof_spearman"].round(3),
            "summarise": res_sum["oof_spearman"].round(3),
            "transfer":  res_xfer["spearman"].round(3),
        })
        print(combined.to_string(index=False))
        out = a.npz.with_name(a.npz.stem + f"_transfer_{a.fmt}.csv")
        combined.to_csv(out, index=False)
        print(f"\nwrote {out}")
        return

    X, y, g, states, meta, vig, fmt_used = _load_vignettes(a.npz, a.fmt)
    res = layerwise_probe(X, y, g)
    rhos = res["oof_spearman"].to_numpy()
    if np.isnan(rhos).all():
        sys.exit("All layers returned NaN rho — not enough data to probe.")
    best = int(np.nanargmax(rhos))
    p_val, count = permutation_p(X, y, g, res.loc[best, "oof_spearman"], n_perm=a.n_perm)
    p_str = f"p < {1/(a.n_perm+1):.4g}" if count == 0 else f"p={p_val:.3f}"

    has_self = (~vig).any()
    if has_self:
        probe = make_probe().fit(X[:, best, :].astype(np.float64), y)
        self_read = float(probe.predict(states[~vig][:, best, :].astype(np.float64))[0])
        logit_path = a.npz.with_name(a.npz.stem + "_logit_ratings.jsonl")
        verbal_adj = logit_vignette_adjusted_self(logit_path, fmt_used)

    out = a.npz.with_name(a.npz.stem + f"_probe_{fmt_used}.csv")
    res.to_csv(out, index=False)
    print(res.round(3).to_string(index=False))
    print(f"\nbest layer {best}: out-of-fold rho={res.loc[best, 'oof_spearman']:.3f} "
          f"({p_str}, {a.n_perm} perms)")
    if has_self:
        print(f"probe read-out (1..5 scale):          {self_read:.2f}  (exploratory)")
        if verbal_adj is not None:
            print(f"verbal (logit), vignette-adjusted:   {verbal_adj:.2f}")
    else:
        print("(no self item in this npz — skipping self read-out)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
