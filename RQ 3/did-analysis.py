"""
Research question:
PRE-TREATMENT TREND: ORS ANALYSIS
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

df = pd.read_csv("../All_Companies_Database.csv")

df['Year founded'] = pd.to_numeric(df['Year founded'], errors='coerce')
df_clean = df[df['Year founded'] > 2010].copy()

target_types = ['Specialized (focused on alternative proteins)', 'Diversified']
df_clean = df_clean[df_clean['Company type'].isin(target_types)]

df_clean['DealSize'] = pd.to_numeric(df_clean['DealSize'].replace('-', np.nan), errors='coerce')
df_clean['is_specialized'] = np.where(
    df_clean['Company type'] == 'Specialized (focused on alternative proteins)', 1, 0
)
df_clean['DealYear'] = pd.to_datetime(df_clean['DealDate'], errors='coerce').dt.year
df_clean['DealYear'] = df_clean['DealYear'].fillna(df_clean['Year'])

df_analysis = df_clean.dropna(subset=['DealSize', 'DealYear'])

total_funding = df_analysis.groupby('Company type')['DealSize'].sum()
mean_funding = df_analysis.groupby('Company type')['DealSize'].mean()

print(total_funding)
print(mean_funding)

model = smf.ols("DealSize ~ DealYear + is_specialized + DealYear:is_specialized", data=df_analysis).fit()

print(model.summary())