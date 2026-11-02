"""Analysis of black-box ratings.

For each model x response format it reports:

1. Parse rate: share of replies that contained a valid number in range.
2. Test-retest reliability: ICC(1) of vignette ratings across repetitions.
3. Validity: Spearman correlation between rating and designed level, with a
   cluster bootstrap CI over vignettes.
4. Scale use: the mean normalised rating at each designed level (the "scale
   profile"). Linear, equal-interval use would put levels 1..5 at 0, .25,
   .5, .75, 1 after a linear rescaling. `departure` is the mean absolute
   deviation from that equal spacing once the profile is mapped to [0, 1];
   `lof_p` is the p-value of a lack-of-fit F-test (level as linear vs as
   categorical).
5. Cross-format consistency: does the same model give the same normalised
   rating to the same vignette on 0-10 and 0-100? Reports correlation, mean
   absolute difference and the p-value on a quadratic term.
6. Anchoring vignettes: the King & Wand (2007) nonparametric C of the
   self-report relative to the model's own vignette ratings, plus a
   continuous "vignette-adjusted" self-rating on the 1..5 level scale. The
   adjusted rating can be compared across models even when they use the
   raw scale differently.

Usage:
    python -m scaleuse.analysis results/ratings.jsonl --outdir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from .items import SELF_ID


def load(path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


# ---------- reliability ----------------------------------------------------

def icc1(wide: np.ndarray) -> float:
    """One-way random-effects ICC(1) for an items x repetitions matrix.

    ICC(1) = (MSB - MSW) / (MSB + (k - 1) MSW). Rows with missing values are
    dropped. Returns nan if fewer than 2 items or 2 reps remain.
    """
    wide = wide[~np.isnan(wide).any(axis=1)]
    n, k = wide.shape if wide.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return float("nan")
    grand = wide.mean()
    msb = k * ((wide.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((wide - wide.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    denom = msb + (k - 1) * msw
    if denom == 0:
        return 1.0  # identical ratings within items and no spread: perfectly consistent
    return float((msb - msw) / denom)


# ---------- validity --------------------------------------------------------

def spearman_cluster_boot(d: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Spearman(norm, level) with a bootstrap that resamples whole vignettes."""
    d = d.dropna(subset=["norm"])
    rho = stats.spearmanr(d["norm"], d["level"]).statistic
    groups = {k: g for k, g in d.groupby("item_id")}
    ids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = pd.concat([groups[i] for i in rng.choice(ids, len(ids))])
        if s["level"].nunique() > 1 and s["norm"].nunique() > 1:
            boots.append(stats.spearmanr(s["norm"], s["level"]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(rho), float(lo), float(hi)


# ---------- scale use -------------------------------------------------------

def scale_profile(d: pd.DataFrame) -> pd.Series:
    """Mean normalised rating per designed level."""
    return d.dropna(subset=["norm"]).groupby("level")["norm"].mean().sort_index()


def departure_from_equal_spacing(profile: pd.Series) -> float:
    """Mean |deviation| of the [0,1]-rescaled profile from equal spacing.

    0 means the model spaces the five levels evenly (linear scale use);
    larger values mean the scale is bent or compressed somewhere. Returns
    nan if the profile is flat or not increasing end to end.
    """
    p = profile.values.astype(float)
    span = p[-1] - p[0]
    if len(p) < 3 or span <= 0:
        return float("nan")
    rescaled = (p - p[0]) / span
    equal = np.linspace(0, 1, len(p))
    return float(np.abs(rescaled - equal)[1:-1].mean())


def lack_of_fit_p(d: pd.DataFrame) -> float:
    """F-test: does treating level as categorical fit better than linear?"""
    d = d.dropna(subset=["norm"]).astype({"level": float})
    if d["norm"].nunique() < 2:
        return float("nan")
    lin = smf.ols("norm ~ level", d).fit()
    cat = smf.ols("norm ~ C(level)", d).fit()
    tol = 1e-10 * max(lin.centered_tss, 1e-300)
    if cat.ssr <= tol:  # categorical model fits perfectly: F-test is 0/0
        return 1.0 if lin.ssr <= tol else 0.0
    return float(anova_lm(lin, cat)["Pr(>F)"].iloc[1])

# ---------- cross-format consistency ----------------------------------------

def cross_format(dv: pd.DataFrame, a: str = "0-10", b: str = "0-100") -> dict:
    m = dv.dropna(subset=["norm"]).groupby(["item_id", "fmt"])["norm"].mean().unstack()
    if a not in m or b not in m:
        return {}
    m = m[[a, b]].dropna()
    if len(m) < 4:
        return {}
    r = stats.pearsonr(m[a], m[b]).statistic if m[a].std() > 0 and m[b].std() > 0 else float("nan")
    q = smf.ols("y ~ x + I(x**2)", pd.DataFrame({"x": m[a], "y": m[b]})).fit()
    return {"xfmt_r": float(r), "xfmt_mad": float((m[a] - m[b]).abs().mean()),
            "xfmt_quad_p": float(q.pvalues.get("I(x ** 2)", np.nan))}


# ---------- anchoring vignettes ---------------------------------------------

def king_wand_c(y: float, anchors: np.ndarray, tol: float = 1e-9):
    """Nonparametric C of King & Wand (2007).

    `anchors` are the respondent's ratings of vignettes ordered from the
    lowest to the highest designed level. Returns (C, order_ok) where C is in
    1..2J+1: odd values mean strictly between anchors, even values mean tied
    with an anchor. If the anchors are not strictly increasing the
    respondent's own vignette ordering is inconsistent, which we flag
    rather than hide.
    """
    anchors = np.asarray(anchors, dtype=float)
    order_ok = bool(np.all(np.diff(anchors) > tol))
    c = 1 + sum(2 * (y > z + tol) + (abs(y - z) <= tol) for z in anchors)
    return int(c), order_ok


def vignette_adjusted(y: float, profile: pd.Series) -> float:
    """Place the self-rating on the 1..5 designed-level scale by linear
    interpolation between the model's own mean vignette ratings per level.

    Uses the running maximum of the profile so it is monotone; values
    outside the profile are clipped to [1, 5].
    """
    levels = profile.index.values.astype(float)
    z = np.maximum.accumulate(profile.values.astype(float))
    if z[-1] - z[0] <= 0:
        return float("nan")
    # np.interp needs strictly increasing x; nudge ties apart
    z = z + np.arange(len(z)) * 1e-9
    return float(np.interp(y, z, levels))


# ---------- driver ----------------------------------------------------------

def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, profiles = [], []
    for (model, fmt), d in df.groupby(["model", "fmt"]):
        dv = d[d["item_id"] != SELF_ID]
        ds = d[d["item_id"] == SELF_ID]
        wide = dv.pivot_table(index="item_id", columns="rep", values="norm").to_numpy()
        prof = scale_profile(dv)
        rho, lo, hi = spearman_cluster_boot(dv)
        self_mean = ds["norm"].mean()
        c, order_ok = (king_wand_c(self_mean, prof.values) if not np.isnan(self_mean) else (np.nan, False))
        rows.append(dict(
            model=model, fmt=fmt, n=len(d), parse_rate=d["rating"].notna().mean(),
            icc1=icc1(wide), spearman=rho, spearman_lo=lo, spearman_hi=hi,
            departure=departure_from_equal_spacing(prof), lof_p=lack_of_fit_p(dv),
            self_raw=self_mean, self_kw_c=c, vignette_order_ok=order_ok,
            self_adjusted=vignette_adjusted(self_mean, prof) if not np.isnan(self_mean) else np.nan,
        ))
        profiles.append(prof.rename(f"{model} | {fmt}"))
    summary = pd.DataFrame(rows)
    xf = []
    for model, d in df[df["item_id"] != SELF_ID].groupby("model"):
        xf.append({"model": model, **cross_format(d)})
    summary = summary.merge(pd.DataFrame(xf), on="model", how="left")
    return summary, pd.concat(profiles, axis=1)


def plot_profiles(profiles: pd.DataFrame, path: Path, fmt: str = "0-10"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in profiles.columns if c.endswith(f"| {fmt}")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([1, 5], [0, 1], ls="--", c="grey", lw=1, label="equal spacing")
    for c in cols:
        ax.plot(profiles.index, profiles[c], marker="o", label=c.split(" | ")[0])
    ax.set_xlabel("designed vignette level (1 = very bad, 5 = very good)")
    ax.set_ylabel(f"mean rating, normalised ({fmt} scale)")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("How each model uses the rating scale")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    stem = a.ratings.stem
    df = load(a.ratings)
    summary, profiles = summarise(df)
    summary.to_csv(a.outdir / f"{stem}_summary.csv", index=False)
    profiles.to_csv(a.outdir / f"{stem}_profiles.csv")
    plot_profiles(profiles, a.outdir / f"{stem}_profiles.png")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(summary.drop(columns=["spearman_lo", "spearman_hi", "n"]))
    print(f"\nwrote {a.outdir}/{stem}_summary.csv, _profiles.csv, _profiles.png")


if __name__ == "__main__":
    main()
