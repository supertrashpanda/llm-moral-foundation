import os
os.environ["OPENAI_API_KEY"] = "EMPTY"
import sys
import run_grid_qwen as rg

v = rg.load_validation()
ex = rg.load_examples()

if len(sys.argv) > 1 and sys.argv[1] == "test":
    # small sample for testing
    print(rg.run_cell(v.head(5), "main", 0, save=False)[rg.LABELS + ["problem"]])
else:
    rg.run_grid(v, examples=ex)    