
import pandas as pd
import os
df_pq = pd.read_parquet("data/winequality.parquet")
print(df_pq.dtypes)

"""antoniasturm@Antonias-MacBook-Pro wine_qual % uv run compare.py
fixed_acidity           float64
volatile_acidity        float64
citric_acid             float64
residual_sugar          float64
chlorides               float64
free_sulfur_dioxide     float64
total_sulfur_dioxide    float64
density                 float64
pH                      float64
sulphates               float64
alcohol                 float64
quality                   int64
wine_color                  str
dtype: object
"""
