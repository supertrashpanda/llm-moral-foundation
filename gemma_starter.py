import os, sys
os.environ["OPENAI_API_KEY"] = "EMPTY"
import run_grid_gemma as rg

v = rg.load_validation()
ex = rg.load_examples()

if len(sys.argv) > 1 and sys.argv[1] == "test":
    pred = rg.run_cell(v.head(5), "main", 0, save=False)
    pred.to_csv("test5_gemma_pred.csv", index=False)
    print(pred[rg.LABELS + ["problem"]].to_string())
else:
    rg.run_grid(v, examples=ex)
