"""
Research question: Did COVID-19 affect the likelihood of receiving funding for specialized alternative protein firms relative to diversified firms?
IV: Company type (Diversified vs. Specialized)
DV: Raised_YN (whether the firm successfully raised VC)
"""
import pandas as pd
import statsmodels.formula.api as smf

#Upload data
df = pd.read_csv('../All_Companies_Database.csv')

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

#Prepare data for regression
data['outcome'] = data['Raised_Binary']
data['treatment'] = data['Company_Type_Dummy']
data['post'] = (data['Year'] >= 2020).astype(int)
data['treatment_post'] = data['treatment'] * data['post']

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
plt.plot([0, 1], means.loc[0], 'b-o', label='Control Group (Diversified)', linewidth=2)
plt.plot([0, 1], means.loc[1], 'r-o', label='Treatment Group (Specialized)', linewidth=2)
plt.xlabel('Time period')
plt.ylabel('Probability of receiving funding')
plt.title('Difference-in-Differences Results (Pre vs Post 2020)')
plt.legend()
plt.xticks([0, 1], ['Pre-2020', 'Post-2020'])
plt.savefig('did-visualization.png')
plt.show()

