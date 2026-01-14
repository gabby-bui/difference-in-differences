import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Upload data
df = pd.read_csv("../All_Companies_Database.csv")

#Clean data
def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
        if x.strip() in ['-', '']: return None
        try: return float(x)
        except: return None
    return x

def categorize_b2b(val):
    if pd.isna(val): return None
    if 'B2B' in val and 'B2C' in val: return None
    if 'B2B' in val: return 1  # Treatment Group
    if 'B2C' in val: return 0  # Control Group
    return None

def clean_year(y):
    try: return int(y)
    except: return None

df['RaisedToDate_cleaned'] = df['RaisedToDate'].apply(clean_currency)
df['Business_Model_Dummy'] = df['B2B/B2C'].apply(categorize_b2b)
df['Year_founded_clean'] = df['Year founded'].apply(clean_year)

#Filter cohort (to only have companies founded in/before 2015)
data = df[(df['Year_founded_clean'] <= 2015) & (df['Business_Model_Dummy'].notnull()) & (df['RaisedToDate_cleaned'].notnull())].copy()

#Set up variables
data['outcome'] = np.log1p(data['RaisedToDate_cleaned'])
data['treatment'] = data['Business_Model_Dummy']
data['time'] = data['Year']

#Calculate mean outcomes for each group and time period
means = data.groupby(['treatment', 'time'])['outcome'].mean().unstack()

#Plot pre-treatment trends
pre_treatment_years = [y for y in means.columns if y < 2020 and y >= 2012]
plt.figure(figsize=(10, 6))
plt.plot(pre_treatment_years, means.loc[0, pre_treatment_years], 'b-o', label='Control Group (B2C)')
plt.plot(pre_treatment_years, means.loc[1, pre_treatment_years], 'r-o', label='Treatment Group (B2B)')
plt.xlabel('Time (Year)')
plt.ylabel('Mean Outcome (Log Capital Raised)')
plt.title('Pre-treatment Trends')
plt.legend()
plt.savefig('pre-treatment-trends.png')
plt.show()

#Test for parallel trends
import statsmodels.formula.api as smf

pre_treatment_data = data[(data['time'] < 2020) & (data['time'] >= 2012)]
model = smf.ols('outcome ~ treatment * time', data=pre_treatment_data).fit()
print(model.summary())
