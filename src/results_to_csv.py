import pandas as pd

results = pd.read_parquet("runs/simulation_results100_new_mp.parquet")
results.to_csv("runs/simulation_results100_new_mp.csv", index=False)
