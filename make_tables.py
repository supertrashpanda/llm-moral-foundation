"""
Build Table 1 (baseline vs. humans) and Table 2 (each condition vs. baseline) for ONE model
from the files written by run_grid_openai.py. Needs no API key.

In a notebook (same folder as this file):
    import make_tables as mt
    t1 = mt.table1()      # baseline measurement quality
    t2 = mt.table2()      # condition-level results
"""
import itertools
import numpy as np
import pandas as pd
import krippendorff

# ---------------------------------------------------------------- settings
OUT_DIR = "runs/gpt-6-luna"     # folder with {variant}_{shots}shot_validation.csv files
HUMAN = ("https://raw.githubusercontent.com/supertrashpanda/llm-moral-rhetoric-tutorial/"
         "refs/heads/main/small_test_set.csv")   # the SAME validation set the LLM runs used
CODERS = ["Yunfan", "Wiktoria", "Carlos"]
LABELS = ["moral_rhetoric", "care", "fairness", "loyalty", "authority", "sanctity"]
FOUNDATIONS = LABELS[1:]
HUMAN_COL = {"moral_rhetoric": "moral", **{f: f for f in FOUNDATIONS}}   # label name in the human file
VARIANTS = ["main", "paraphrase", "reverse"]
SHOTS = [0, 2, 5]
BASELINE = ("main", 0)
N_BOOT = 1000
SEED = 502


def alpha(*codes):
    """Nominal Krippendorff's alpha for 2+ coders (arrays of 0/1, NaN = missing)."""
    data = np.vstack([np.asarray(c, dtype=float) for c in codes])
    vals = data[~np.isnan(data)]
    if len(np.unique(vals)) < 2:
        return np.nan
    return krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")


def load_humans(path=None):
    """path=None uses mt.HUMAN as it is set now (so mt.HUMAN = "..." in a notebook works)."""
    h = pd.read_csv(path or HUMAN)
    for l in LABELS:
        cols = [f"{HUMAN_COL[l]}_{c}" for c in CODERS]
        h[f"{l}_majority"] = (h[cols].sum(axis=1) >= 2).astype(int)
    return h


def _check_overlap(h, c, what):
    """c: a results DataFrame (with para_id) or a collection of para_ids.
    Returns the para_ids shared with the human file; raises a clear error if there are none."""
    ids = set(c["para_id"]) if isinstance(c, pd.DataFrame) else set(c)
    shared = set(h["para_id"]) & ids
    if not shared:
        raise ValueError(
            f"No para_id in {what} matches the human-coded file ({HUMAN}).\n"
            f"  human file ids: {h['para_id'].head(2).tolist()}\n"
            f"  {what} ids:  {sorted(ids)[:2]}\n"
            "Set mt.HUMAN to the validation set the LLM runs were made on.")
    if len(shared) < len(ids):
        print(f"note: {len(ids) - len(shared)} paragraph(s) in {what} are not in the human file and are skipped")
    return shared


def load_cell(variant, shots, rep=1):
    stem = f"{OUT_DIR}/{variant}_{shots}shot_validation" + ("" if rep == 1 else f"_rep{rep}")
    return pd.read_csv(f"{stem}.csv")[["para_id"] + LABELS]


def _summary_rows(per_label):
    """Add a 'foundations (mean)' row: the mean over the five foundations."""
    df = pd.DataFrame(per_label).set_index("label")
    df.loc["foundations (mean)"] = df.loc[FOUNDATIONS].mean(numeric_only=True)
    return df


# ---------------------------------------------------------------- Table 1
def table1(humans=None):
    """Baseline condition vs. humans, one row per label, plus the mean over foundations."""
    h = load_humans() if humans is None else humans
    base = load_cell(*BASELINE)
    _check_overlap(h, base, "main_0shot_validation.csv")
    m = h.merge(base, on="para_id")
    rows = []
    for l in LABELS:
        llm = m[l]
        coders = [m[f"{HUMAN_COL[l]}_{c}"] for c in CODERS]
        hh = [alpha(a, b) for a, b in itertools.combinations(coders, 2)]
        rows.append({
            "label": l,
            "n": int(llm.notna().sum()),
            "alpha_vs_majority": alpha(m[f"{l}_majority"], llm),
            "alpha_vs_coders_mean": np.nanmean([alpha(c, llm) for c in coders]),
            "human_human_alpha_mean": np.nanmean(hh),
            "human_3coder_alpha": alpha(*coders),
            "llm_share_present": llm.mean(),
            "majority_share_present": m[f"{l}_majority"].mean(),
        })
    return _summary_rows(rows).round(3)


# ---------------------------------------------------------------- Table 2
def _boot_delta(y, a, b, rng):
    """Paired bootstrap of alpha(y, b) - alpha(y, a), resampling paragraphs.
    y, a, b: dict label -> array. Returns dict label -> (low, high), incl. 'foundations (mean)'."""
    n = len(next(iter(y.values())))
    draws = {l: [] for l in LABELS + ["foundations (mean)"]}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        d = {l: alpha(y[l][i], b[l][i]) - alpha(y[l][i], a[l][i]) for l in LABELS}
        for l in LABELS:
            draws[l].append(d[l])
        draws["foundations (mean)"].append(np.nanmean([d[f] for f in FOUNDATIONS]))
    return {l: tuple(np.nanpercentile(v, [2.5, 97.5])) for l, v in draws.items()}


def table2(humans=None, labels=("moral_rhetoric", "foundations (mean)")):
    """One row per condition, compared with the baseline on the paragraphs every condition coded.
    First row = baseline run 2 (noise floor). Columns: alpha with baseline, alpha vs majority,
    change vs baseline (paired-bootstrap 95% CI), share coded present.
    labels: which rows to show; use mt.LABELS + ["foundations (mean)"] for all."""
    h = load_humans() if humans is None else humans
    cells = {(v, k): load_cell(v, k) for v in VARIANTS for k in SHOTS}
    RERUN = ("main", "0 (run 2)")
    try:
        cells[RERUN] = load_cell("main", 0, rep=2)
    except FileNotFoundError:
        print("note: no baseline run 2 file found; skipping the noise-floor row")
    common = set.intersection(*[set(c["para_id"]) for c in cells.values()])
    common &= _check_overlap(h, cells[BASELINE], "main_0shot_validation.csv")
    # drop paragraphs with a parse/API error in any condition
    for c in cells.values():
        common -= set(c.loc[c[LABELS].isna().any(axis=1), "para_id"])
    h = h[h["para_id"].isin(common)].sort_values("para_id").reset_index(drop=True)
    cells = {key: c[c["para_id"].isin(common)].sort_values("para_id").reset_index(drop=True)
             for key, c in cells.items()}
    y = {l: h[f"{l}_majority"].to_numpy(float) for l in LABELS}
    a = {l: cells[BASELINE][l].to_numpy(float) for l in LABELS}
    rng = np.random.default_rng(SEED)

    order = ([RERUN] if RERUN in cells else []) + [(v, k) for v in VARIANTS for k in SHOTS]
    out = []
    for key in order:
        b = {l: cells[key][l].to_numpy(float) for l in LABELS}
        is_base = key == BASELINE
        ci = {} if is_base else _boot_delta(y, a, b, rng)
        per_label = [{
            "label": l,
            "alpha_with_baseline": np.nan if is_base else alpha(a[l], b[l]),
            "alpha_vs_majority": alpha(y[l], b[l]),
            "delta_vs_baseline": np.nan if is_base else alpha(y[l], b[l]) - alpha(y[l], a[l]),
            "share_present": b[l].mean(),
        } for l in LABELS]
        df = _summary_rows(per_label)
        name = "main x 0-shot (run 2)" if key == RERUN else f"{key[0]} x {key[1]}-shot" + (" (baseline)" if is_base else "")
        for l in labels:
            lo, hi = ci.get(l, (np.nan, np.nan))
            out.append({"condition": name, "label": l, **df.loc[l].to_dict(),
                        "delta_ci_low": lo, "delta_ci_high": hi})
    t = pd.DataFrame(out).set_index(["condition", "label"])
    t = t[["alpha_with_baseline", "alpha_vs_majority", "delta_vs_baseline",
           "delta_ci_low", "delta_ci_high", "share_present"]].round(3)
    print(f"Table 2 uses the {len(common)} paragraphs coded (without errors) in every condition.")
    return t