"""
Homework 19/01 - 26/01 (Data Team)
Value Chain Analysis
"""
import pandas as pd

# Load data
file = ('../All_Companies_Database.csv')
df = pd.read_csv(file)

#Identify which companies ever raised funding
company_funding = df.groupby('Company')['Raised_YN'].apply(
    lambda x: 'Y' if 'Y' in x.values else 'N'
).reset_index()
company_funding.columns = ['Company', 'Ever_Raised']

print(f"\nCompanies that ever raised funding: {(company_funding['Ever_Raised'] == 'Y').sum():,}")
print(f"Companies that never raised funding: {(company_funding['Ever_Raised'] == 'N').sum():,}")

#Get Technology Focus for each company (take most recent year)
latest_year = df.groupby('Company')['Year'].max().reset_index()
latest_year.columns = ['Company', 'Latest_Year']

company_tech = df.merge(latest_year, on='Company')
company_tech = company_tech[company_tech['Year'] == company_tech['Latest_Year']]
company_tech = company_tech[['Company', 'Technology Focus']].drop_duplicates()

#Merge funding status with technology focus
company_analysis = company_tech.merge(company_funding, on='Company')

#Calculate % with funding for each Technology Focus
tech_results = []

for idx, row in company_analysis.iterrows():
    company = row['Company']
    ever_raised = row['Ever_Raised']
    tech_focus = row['Technology Focus']

    #Split multiple tech focuses
    if pd.notna(tech_focus) and tech_focus != '-':
        techs = [t.strip() for t in str(tech_focus).split(',')]
        for tech in techs:
            tech_results.append({
                'Company': company,
                'Technology_Focus': tech,
                'Ever_Raised': ever_raised
            })

tech_df = pd.DataFrame(tech_results)

#Calculate percentages
tech_summary = tech_df.groupby('Technology_Focus').agg(
    total_companies=('Company', 'nunique'),
    funded_companies=('Ever_Raised', lambda x: (x == 'Y').sum())
).reset_index()

tech_summary['pct_funded'] = (tech_summary['funded_companies'] / tech_summary['total_companies'] * 100).round(2)
tech_summary = tech_summary.sort_values('pct_funded', ascending=False)

print(tech_summary.to_string(index=False))

#Save to CSV
tech_summary.to_csv('../pct_funded_by_tech_focus.csv', index=False)

funding_events = df[df['Raised_YN'] == 'Y'].copy()

print(f"\nTotal funding events in dataset: {len(funding_events):,}")

tech_funding_counts = []

for idx, row in funding_events.iterrows():
    tech_focus = row['Technology Focus']

    if pd.notna(tech_focus) and tech_focus != '-':
        techs = [t.strip() for t in str(tech_focus).split(',')]
        for tech in techs:
            tech_funding_counts.append({
                'Technology_Focus': tech,
                'Company': row['Company'],
                'Year': row['Year']
            })

tech_funding_df = pd.DataFrame(tech_funding_counts)

tech_funding_summary = tech_funding_df.groupby('Technology_Focus').size().reset_index(name='total_funding_events')
tech_funding_summary = tech_funding_summary.sort_values('total_funding_events', ascending=False)

print(tech_funding_summary.head(20).to_string(index=False))

#Save to CSV
tech_funding_summary.to_csv('../funding_by_tech_focus.csv', index=False)