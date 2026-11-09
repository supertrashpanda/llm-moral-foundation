"""
Moral foundation coding grid with an OpenAI model, in the same format as llm_moral_coding.ipynb.

Design: prompt variant {main, paraphrase, reverse} x shots {0, 2, 5} = 9 cells.
Primary cell (pre-specified) = main x 0-shot; the other 8 are robustness checks.

- main        : your CODEBOOK and make_prompt, word for word
- paraphrase  : same instructions, reworded label definitions (DEFINITIONS_PARAPHRASE)
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
DATA = "https://raw.githubusercontent.com/supertrashpanda/llm-moral-rhetoric-tutorial/refs/heads/main/50_results2.csv"
EXAMPLES_FILE = "fewshot_examples.csv"   # only needed for 2- and 5-shot cells
OUT_DIR = f"runs/{MODEL}"

LABELS = ["moral_rhetoric", "care", "fairness", "loyalty", "authority", "sanctity"]
FOUNDATIONS = LABELS[1:]
VARIANTS = ["main", "paraphrase", "reverse"]
SHOTS = [0, 2, 5]
PRIMARY = ("main", 0)

client = OpenAI(max_retries=5)

# ---------------------------------------------------------------- codebook
# Sections 1-2 state the coding direction, so the reverse variant has its own version.
INSTRUCTIONS = {
"normal": '''
1. TASK
You're an expert political scientist analyzing U.S. congressional floor speeches. You need to decide whether a paragraph from the speeches uses moral rhetoric, and if it does, which moral foundations it draws on.
Code six labels, each as 1 (present) or 0 (absent): MORAL RHETORIC, CARE, FAIRNESS, LOYALTY, AUTHORITY, SANCTITY.

2. CODING APPROACH
* The unit of analysis is the paragraph. Read the full paragraph before coding.
* Code each label for presence (1) or absence (0).
* Code a label as 1 if the paragraph expresses its meaning at least once.
* A paragraph can be coded for several moral foundations at the same time.

Procedure:
1. Read the full paragraph.
2. Code MORAL RHETORIC as 1 or 0.
3. If MORAL RHETORIC is 1, code each of the five moral foundations as 1 or 0. At least one foundation must be 1.
4. If MORAL RHETORIC is 0, all five moral foundations must be 0.
''',
"reverse": '''
1. TASK
You're an expert political scientist analyzing U.S. congressional floor speeches. You need to decide whether a paragraph from the speeches uses moral rhetoric, and if it does, which moral foundations it draws on.
Code six labels, each as 0 (present) or 1 (absent): MORAL RHETORIC, CARE, FAIRNESS, LOYALTY, AUTHORITY, SANCTITY.

2. CODING APPROACH
* The unit of analysis is the paragraph. Read the full paragraph before coding.
* Code each label for presence (0) or absence (1).
* Code a label as 0 if the paragraph expresses its meaning at least once.
* A paragraph can be coded for several moral foundations at the same time.

Procedure:
1. Read the full paragraph.
2. Code MORAL RHETORIC as 0 or 1.
3. If MORAL RHETORIC is 0, code each of the five moral foundations as 0 or 1. At least one foundation must be 0.
4. If MORAL RHETORIC is 1, all five moral foundations must be 1.
''',
}

DEFINITIONS = '''
3. LABELS

Label: MORAL RHETORIC
Definition: The speaker explicitly or implicitly judges an action, actor, or policy as morally right/wrong, good/evil, just/unjust, or as a duty/violation — i.e., the justification appeals to values, not only to consequences.
If the paragraph is not discussing something being right or wrong, in other words if it is a rational, abstract description of a state of the matter, it should not be coded as moral.
Negative example: "When the United States hesitates or sends mixed signals, we do not create stability; we invite aggression. Let's be clear about that. History is replete with examples of those who hesitate or send mixed signals. Therefore, American leadership remains the cornerstone of global security."


Label: CARE
Definition: The paragraph emphasizes
* positive: protection, compassion, helping, saving, or defending others, OR
* negative: harm, suffering, damage, war, killing.
Examples:
* repeated references to helping families, protecting citizens
* emphasis on supporting vulnerable groups (e.g., children, veterans, elderly)
* claims that a policy will hurt working families or cause unnecessary suffering
* framing an issue in terms of human cost (e.g., "people are struggling," "lives are at risk")

Label: FAIRNESS
Definition: The paragraph emphasizes
* positive: justice, equality, fairness, honesty, OR
* negative: injustice, inequality, corruption, dishonesty.
Examples:
* calls for equal opportunity or fair treatment
* claims that a system is rigged, unfair, or stacked against certain groups
* references to discrimination, inequality, or corruption in institutions

Label: LOYALTY
Definition: The paragraph emphasizes
* positive: unity, patriotism, national identity, solidarity, support, nation, family, OR
* negative: betrayal (in-group), division, threats to the group, abandonment, rebellion (against authority).
Examples:
* appeals to national unity
* references to patriotism or national pride
* criticism of actions that divide the nation or undermine our unity
* accusations that political actors have betrayed the American people

Label: AUTHORITY
Definition: The paragraph emphasizes
* positive: law, order, institutional respect, duty, obligation, OR
* negative: illegitimacy, abuse of power, betrayal (against your role), rebellion (against authority), chaos, disorder.
Examples:
* emphasis on upholding the Constitution, rule of law, or democratic norms
* calls to respect institutions (courts, Congress, law enforcement, etc.)
* claims that leaders are abusing power or acting unlawfully
* references to lawlessness, constitutional violations, or breakdown of order

Label: SANCTITY
Definition: The paragraph emphasizes
* positive: dignity, moral integrity, sacred values, purity, preservation, serenity, OR
* negative: moral decay, degradation, disgust, repulsiveness, dirtiness, revoltingness.
Examples:
* appeals to human dignity or moral responsibility
* framing issues as matters of moral integrity or core values
* references to societal decline, or erosion of values
'''

# TODO: reword the definitions (same substance) before running the "paraphrase" cells.
DEFINITIONS_PARAPHRASE = DEFINITIONS


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
    if variant == "reverse":
        answer = ('Answer in JSON format. First give the evidence: for each label, a short quote from the '
                  'paragraph that shows it, or "" if the label is 1. Then give the labels.\n'
                  '{"evidence": {"moral_rhetoric": "quote or empty", "care": "quote or empty", "fairness": '
                  '"quote or empty", "loyalty": "quote or empty", "authority": "quote or empty", "sanctity": '
                  '"quote or empty"}, "moral_rhetoric": 0 or 1, "care": 0 or 1, "fairness": 0 or 1, '
                  '"loyalty": 0 or 1, "authority": 0 or 1, "sanctity": 0 or 1}')
    else:
        answer = ('Answer in JSON format. First give the evidence: for each label, a short quote from the '
                  'paragraph that shows it, or "" if the label is 0. Then give the labels.\n'
                  '{"evidence": {"moral_rhetoric": "quote or empty", "care": "quote or empty", "fairness": '
                  '"quote or empty", "loyalty": "quote or empty", "authority": "quote or empty", "sanctity": '
                  '"quote or empty"}, "moral_rhetoric": 1 or 0, "care": 1 or 0, "fairness": 1 or 0, '
                  '"loyalty": 1 or 0, "authority": 1 or 0, "sanctity": 1 or 0}')
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
            client.chat.completions.create(messages=[{"role": "user", "content": prompt}], **PARAMS)
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
            resp = client.chat.completions.create(
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

    pred = pd.DataFrame([{"para_id": pid, "variant": variant, "shots": shots, **o}
                         for pid, o in zip(data["para_id"], outs)])
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
