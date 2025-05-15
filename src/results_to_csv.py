import pandas as pd

results = pd.read_parquet("runs/simulation_results_1000_network_514.parquet")
results.to_csv("runs/simulation_results_1000_network_514.csv", index=False)
