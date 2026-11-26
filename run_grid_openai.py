"""
Moral foundation coding grid with an OpenAI model, in the same format as llm_moral_coding.ipynb.

Design: prompt variant {main, paraphrase, reverse} x shots {0, 2, 5} = 9 cells.
Primary cell (pre-specified) = main x 0-shot; the other 8 are robustness checks.

- main        : your CODEBOOK and make_prompt, word for word
- paraphrase  : same instructions, paraphrases label definitions (DEFINITIONS_PARAPHRASE)
- reverse     : coding scheme flipped (1 = absent, 0 = present); read_answer flips the
                codes back, so every cell is saved on the same scale (1 = present)

Output columns are the notebook's:  para_id, moral_rhetoric, care, fairness, loyalty,
authority, sanctity, evidence_<label> (x6), problem, warnings, raw, model_version
plus: variant, shots.

Usage (Colab):
    from run_grid_openai import *
    validation = load_validation()
    pred = run_cell(validation, "main", 0)          # primary cell only
    all_pred = run_grid(validation)                  # all 9 cells (needs few-shot examples)
"""
import os, json, re, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI, BadRequestError
from tqdm.auto import tqdm

# ---------------------------------------------------------------- settings
MODEL = "gpt-6-luna"
REASONING_EFFORT = "none"      # non-reasoning mode
TEMPERATURE = 0                # deterministic output (dropped if the model rejects it)
SEED = 502
WORKERS = 8                    # parallel requests; set to 1 to run one at a time
DATA = "https://raw.githubusercontent.com/supertrashpanda/llm-moral-rhetoric-tutorial/refs/heads/main/test_full.csv"
EXAMPLES_FILE = "data/fewshot_examples.csv"   # only needed for 2- and 5-shot cells
OUT_DIR = f"runs/{MODEL}"

LABELS = ["moral_rhetoric", "care", "fairness", "loyalty", "authority", "sanctity"]
FOUNDATIONS = LABELS[1:]
VARIANTS = ["main", "paraphrase", "reverse"]
SHOTS = [0, 2, 5]
PRIMARY = ("main", 0)

client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(max_retries=5)
    return client

# ---------------------------------------------------------------- codebook
# Sections 1-2 state the coding direction, so the reverse variant has its own version.

import hashlib
from pathlib import Path
PROMPT_DIR = Path(__file__).parent / "prompt" if "__file__" in globals() else Path("prompt")

def _read(name):
    return (PROMPT_DIR / name).read_text(encoding="utf-8")

INSTRUCTIONS = {"normal": _read("instructions_normal.txt"), "reverse": _read("instructions_reverse.txt")}
DEFINITIONS = _read("definitions_main.txt")
DEFINITIONS_PARAPHRASE = _read("definitions_paraphrase.txt")
ANSWER = {"normal": _read("answer_normal.txt"), "reverse": _read("answer_reverse.txt")}
PROMPT_HASH = hashlib.sha256("".join(sorted(
    p.read_text(encoding="utf-8") for p in PROMPT_DIR.glob("*.txt"))).encode()).hexdigest()[:12]


def codebook(variant="main"):
    instructions = INSTRUCTIONS["reverse" if variant == "reverse" else "normal"]
    definitions = DEFINITIONS_PARAPHRASE if variant == "paraphrase" else DEFINITIONS
    return instructions + definitions


CODEBOOK = codebook("main")   # identical to the notebook's CODEBOOK

# ---------------------------------------------------------------- data
def load_validation(path=DATA):
    return pd.read_csv(path).rename(columns={"moral": "moral_rhetoric"})


def load_examples(path=EXAMPLES_FILE):
    """Few-shot examples, in the order they should be used (first 2 = 2-shot, first 5 = 5-shot).
    Columns: para_id, text, the six LABELS (0/1, 1 = present), optional evidence_<label>."""
    return pd.read_csv(path)

# ---------------------------------------------------------------- prompt (as in the notebook)
def _answer_json(codes, evidence, variant="main"):
    """The answer format shared by the instructions and the few-shot examples."""
    flip = variant == "reverse"
    ev = ", ".join(f'"{l}": {json.dumps(evidence.get(l, ""))}' for l in LABELS)
    cd = ", ".join(f'"{l}": {1 - int(codes[l]) if flip else int(codes[l])}' for l in LABELS)
    return f'{{"evidence": {{{ev}}}, {cd}}}'


def make_examples(df, variant="main"):
    """Few-shot block from adjudicated development-set paragraphs (never the test set).
    Labels in df are always 1 = present; the reverse variant shows them flipped."""
    if df is None or len(df) == 0:
        return ""
    blocks = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        evidence = {l: r[f"evidence_{l}"] for l in LABELS
                    if f"evidence_{l}" in r and int(r[l]) == 1 and isinstance(r[f"evidence_{l}"], str)}
        blocks.append(f'Example {i}:\nParagraph: "{r["text"]}"\nAnswer: {_answer_json(r, evidence, variant)}')
    return "Coded examples:\n\n" + "\n\n".join(blocks) + "\n"


def make_prompt(paragraph, examples="", variant="main"):
    answer = ANSWER["reverse" if variant == "reverse" else "normal"]
    return f'''
{codebook(variant)}
{examples}
Paragraph to code:
"{paragraph}"

{answer}'''



def _norm(s):
    return re.sub(r"\W+", " ", str(s)).lower().strip()


def read_answer(text, paragraph=None, variant="main"):
    """Parse the model's JSON answer and check the coding rules (codes returned with 1 = present).

    Returns a dict with the six labels, 'evidence_<label>' for each label,
    'problem' (a rule violation that makes the codes unusable, else None) and
    'warnings' (evidence issues worth reviewing; codes are still usable)."""
    out = {l: None for l in LABELS}
    out.update({f"evidence_{l}": None for l in LABELS})
    out.update(problem=None, warnings=[])
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        ans = json.loads(m.group(0))
        ev = ans.get("evidence", {}) or {}
        for l in LABELS:
            code = int(ans[l])
            out[l] = 1 - code if variant == "reverse" else code
            out[f"evidence_{l}"] = ev.get(l, "") or ""
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        out["problem"] = "parse_error"
        return out

    n_found = sum(out[f] for f in FOUNDATIONS)
    if out["moral_rhetoric"] == 1 and n_found == 0:
        out["problem"] = "moral_without_foundation"
    elif out["moral_rhetoric"] == 0 and n_found > 0:
        out["problem"] = "foundation_without_moral"

    for l in LABELS:
        quote = out[f"evidence_{l}"]
        if out[l] == 1 and not quote:
            out["warnings"].append(f"{l}: coded 1 without evidence")
        if out[l] == 0 and quote:
            out["warnings"].append(f"{l}: evidence given but coded 0")
        if paragraph is not None and quote and _norm(quote) not in _norm(paragraph):
            out["warnings"].append(f"{l}: evidence not found in paragraph")
    return out

# ---------------------------------------------------------------- call the model
SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "object",
            "properties": {l: {"type": "string"} for l in LABELS},
            "required": LABELS,
            "additionalProperties": False,
        },
        **{l: {"type": "integer", "enum": [0, 1]} for l in LABELS},
    },
    "required": ["evidence", *LABELS],
    "additionalProperties": False,
}

PARAMS = {
    "model": MODEL,
    "temperature": TEMPERATURE,
    "seed": SEED,
    "reasoning_effort": REASONING_EFFORT,
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "moral_coding", "strict": True, "schema": SCHEMA}},
}
DROPPED = []   # parameters the model rejected (e.g. temperature); see check_params()


def check_params(prompt):
    """Send one test request; if the model rejects temperature or seed, drop it and say so."""
    for _ in range(3):
        try:
            get_client().chat.completions.create(messages=[{"role": "user", "content": prompt}], **PARAMS)
            break
        except BadRequestError as e:
            bad = [p for p in ("temperature", "seed") if p in PARAMS and p in str(e)]
            if not bad:
                raise
            for p in bad:
                PARAMS.pop(p)
                DROPPED.append(p)
    if DROPPED:
        print(f"note: {MODEL} rejected {DROPPED}; running without them")
    return DROPPED


def ask_llm(prompt, paragraph=None, variant="main", retries=3):
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}], **PARAMS)
            text = resp.choices[0].message.content
            out = read_answer(text, paragraph, variant)
            out["raw"] = text
            out["model_version"] = resp.model
            return out
        except Exception as e:
            print(f"attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return {"problem": "api_error"}

# ---------------------------------------------------------------- run
def run_cell(validation, variant="main", shots=0, examples=None, save=True):
    """Code every paragraph once with one (variant, shots) cell. No resume logic."""
    ex_block, data = "", validation
    if shots > 0:
        examples = load_examples() if examples is None else examples
        ex = examples.head(shots)
        overlap = set(ex["para_id"]) & set(validation["para_id"])
        if overlap:   # never score a paragraph that is also shown as an example
            print(f"note: {len(overlap)} example paragraph(s) left out of this cell")
            data = validation[~validation["para_id"].isin(overlap)]
        ex_block = make_examples(ex, variant)

    prompts = [make_prompt(t, ex_block, variant) for t in data["text"]]
    if not DROPPED and "temperature" in PARAMS:
        check_params(prompts[0])

    def one(i):
        return ask_llm(prompts[i], data["text"].iloc[i], variant)

    with ThreadPoolExecutor(WORKERS) as pool:
        outs = list(tqdm(pool.map(one, range(len(prompts))), total=len(prompts),
                         desc=f"{variant}, {shots}-shot"))

    pred = pd.DataFrame([{"para_id": pid, "text": text, "variant": variant, "shots": shots,**o}
                     for pid, text, o in zip(data["para_id"], data["text"], outs)])
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        stem = f"{OUT_DIR}/{variant}_{shots}shot_validation"
        pred.to_csv(f"{stem}.csv", index=False)
        with open(f"{stem}_prompt.txt", "w") as f:       # the exact prompt for the first paragraph
            f.write(prompts[0])
        with open(f"{stem}_settings.json", "w") as f:
            json.dump({"variant": variant, "shots": shots, "primary_cell": (variant, shots) == PRIMARY,
                       "params": {k: v for k, v in PARAMS.items() if k != "response_format"},
                       "dropped_params": DROPPED,
                       "model_versions": sorted(pred["model_version"].dropna().unique().tolist()),
                       "n": len(pred), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    return pred


def run_grid(validation, cells=None, examples=None):
    """Run several cells (default: all 9) and stack the results."""
    cells = cells or [(v, k) for v in VARIANTS for k in SHOTS]
    return pd.concat([run_cell(validation, v, k, examples) for v, k in cells], ignore_index=True)
