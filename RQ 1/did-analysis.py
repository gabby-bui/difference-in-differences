"""
Research question: Did the 2020 market shock cause a structural shift in capital allocation, specifically favoring B2B (infrastructure) over B2C (consumer) companies?
2020 market shock:
+ The initial public offering of Beyond Meat
+ The impacts of the COVID-19 global pandemic on geopolitics, food security, and logistics
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#Upload data
df = pd.read_csv('../All_Companies_Database.csv')

#Clean data
def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
        if x.strip() in ['-', '']: return None
        try: return float(x)
        except: return None
    return x

def clean_year(y):
    try: return int(y)
    except: return None

def categorize_b2b(val):
    if pd.isna(val): return None
    if 'B2B' in val and 'B2C' in val: return None
    if 'B2B' in val: return 1  #Treatment group (Dummy = 1)
    if 'B2C' in val: return 0  #Control group (Dummy = 0)
    return None

df['RaisedToDate_cleaned'] = df['RaisedToDate'].apply(clean_currency)
df['Year_founded_clean'] = df['Year founded'].apply(clean_year)
df['Business_Model_Dummy'] = df['B2B/B2C'].apply(categorize_b2b)

#Filter cohort (to only have companies founded in/before 2015)
data = df[(df['Year_founded_clean'] <= 2015) & (df['Business_Model_Dummy'].notnull()) & (df['RaisedToDate_cleaned'].notnull())].copy()

#Prepare data for regression
data['outcome'] = np.log1p(data['RaisedToDate_cleaned'])
data['treatment'] = data['Business_Model_Dummy']
data['post'] = (data['Year'] >= 2020).astype(int)
data['treatment_post'] = data['treatment']*data['post']

#Fit the OLS model
model = smf.ols("outcome ~ treatment + post + treatment_post", data=data).fit()

#Print the results
print(model.summary())

#Extract the DiD estimator (coefficient of treatment_post)
did_estimate = model.params['treatment_post']
print(f"\nEstimated treatment effect: {did_estimate:.4f}")

#Visualize DiD results
import matplotlib.pyplot as plt

#Calculate mean outcomes for each group and time period
means = data.groupby(['treatment', 'post'])['outcome'].mean().unstack()

#Create the plot
plt.figure(figsize=(10, 6))
plt.plot([0, 1], means.loc[0], 'b-o', label='Control Group (B2C)', linewidth=2)
plt.plot([0, 1], means.loc[1], 'r-o', label='Treatment Group (B2B)', linewidth=2)
plt.xlabel('Time Period')
plt.ylabel('Mean Outcome (Log Capital Raised)')
plt.title('Difference-in-Differences Results (Pre vs Post 2020)')
plt.legend()
plt.xticks([0, 1], ['Pre-2020', 'Post-2020'])
plt.savefig('did-visualization.png')
plt.show()
