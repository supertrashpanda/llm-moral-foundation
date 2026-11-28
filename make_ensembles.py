"""
Four 3-model ensembles from the baseline runs (standard prompt, 0-shot).
Each ensemble leaves one model out; a label is 1 if at least 2 of the 3 models code it 1.

    python make_ensembles.py      (run it in the folder with make_tables.py and runs/)

Writes, in runs/ensemble/:
    without_<model>/main_0shot_validation.csv   the ensemble's labels
    table1_ensembles.csv                        Table 1 for the four ensembles
    ensemble_votes.csv                          per paragraph: how many of the 4 models code each label 1
"""
import os
import pandas as pd
import make_tables as mt

# display name -> folder with that model's main_0shot_validation.csv (edit to match yours)
MODELS = {
    "gpt":    "runs/gpt-6-luna",
    "claude": "runs/claude-haiku-5-5",
    "gemma":  "runs/gemma-4-31b-it-w4a16",
    "qwen":   "runs/qwen2.5-32b-awq",
}
mt.HUMAN = ("https://raw.githubusercontent.com/supertrashpanda/llm-moral-rhetoric-tutorial/"
            "refs/heads/main/test_full.csv")
OUT = "runs/ensemble"

# baseline labels of each model; keep paragraphs that every model coded without an error
labels = {}
for name, folder in MODELS.items():
    mt.OUT_DIR = folder
    labels[name] = mt.load_cell("main", 0).set_index("para_id")[mt.LABELS]
common = sorted(set.intersection(*[set(d.dropna().index) for d in labels.values()]))
labels = {n: d.loc[common].astype(int) for n, d in labels.items()}
print(f"{len(common)} paragraphs coded by all four models")

# how many of the four models code each label as 1 (0-4), per paragraph
os.makedirs(OUT, exist_ok=True)
sum(labels.values()).reset_index().to_csv(f"{OUT}/ensemble_votes.csv", index=False)

tables = {}
for left_out in MODELS:
    members = [n for n in MODELS if n != left_out]
    votes = sum(labels[n] for n in members)
    ens = (votes >= 2).astype(int).reset_index()

    folder = f"{OUT}/without_{left_out}"
    os.makedirs(folder, exist_ok=True)
    ens.to_csv(f"{folder}/main_0shot_validation.csv", index=False)

    mt.OUT_DIR = folder
    tables[f"without_{left_out}"] = mt.table1()

t = pd.concat(tables, names=["ensemble", "label"])
t.to_csv(f"{OUT}/table1_ensembles.csv")
print(t[["alpha_vs_majority", "alpha_vs_coders_mean", "human_human_alpha_mean",
         "llm_share_present", "majority_share_present"]].to_string())
