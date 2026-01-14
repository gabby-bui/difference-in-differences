import pandas as pd
import matplotlib.pyplot as plt

#Upload data
df = pd.read_csv("../All_Companies_Database.csv")

#Clean data
def clean_year(y):
    try: return int(y)
    except: return None

def categorize_company_type(val):
    if pd.isna(val) or val == '-': return None
    if 'Specialized' in val: return 1 #Treatment group (Dummy = 1)
    if 'Diversified' in val: return 0 #Control group (Dummy = 0)

def clean_raised_yn(val):
    if pd.isna(val): return None
    if val == 'Y' or val == 'Yes' or val == 1: return 1
    if val == 'N' or val == 'No' or val == 0: return 0
    return None

df['Year_founded_clean'] = df['Year founded'].apply(clean_year)
df['Company_Type_Dummy'] = df['Company type'].apply(categorize_company_type)
df['Raised_Binary'] = df['Raised_YN'].apply(clean_raised_yn)

#Filter cohort (to only have companies founded in/before 2015)
data = df[(df['Year_founded_clean'] <= 2015) & (df['Company_Type_Dummy'].notnull()) & (df['Raised_Binary'].notnull())].copy()

#Set up variables
data['outcome'] = data['Raised_Binary']
data['treatment'] = data['Company_Type_Dummy']
data['time'] = data['Year']

#Calculate mean outcomes for each group and time period
means = data.groupby(['treatment', 'time'])['outcome'].mean().unstack()

#Plot pre-treatment trends
pre_treatment_years = [y for y in means.columns if y < 2020 and y >= 2012]
plt.figure(figsize=(10, 6))
plt.plot(pre_treatment_years, means.loc[0, pre_treatment_years], 'b-o', label='Control Group (B2C)')
plt.plot(pre_treatment_years, means.loc[1, pre_treatment_years], 'r-o', label='Treatment Group (B2B)')
plt.xlabel('Time period')
plt.ylabel('Probability of receiving funding')
plt.title('Pre-treatment Trends')
plt.legend()
plt.savefig('pre-treatment-trends.png')
plt.show()

#Test for parallel trends
import statsmodels.formula.api as smf

pre_treatment_data = data[(data['time'] < 2020) & (data['time'] >= 2012)]
model = smf.ols('outcome ~ treatment * time', data=pre_treatment_data).fit()
print(model.summary())
