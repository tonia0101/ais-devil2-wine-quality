
import pandas as pd
import os
df_pq = pd.read_parquet("data/winequality.parquet")
print(df_pq.dtypes)