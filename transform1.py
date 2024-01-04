import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import gradient
from seaborn import saturate
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
from pylab import rcParams

df = pd.read_csv('insurance_claims.csv')   #importing csv file

# Setting up values in place of Y and N to 1 and 0
df['fraud_reported'].replace(to_replace='Y', value=1, inplace=True)
df['fraud_reported'].replace(to_replace='N',  value=0, inplace=True)
pd.set_option('display.max_columns', None)
print(df.head(20))

df[['insured_zip']] = df[['insured_zip']].astype(object)
print(df.describe())    #summary statistics of a DataFrame or a Series.
print(df.describe(include='all'))

# Figure 1

fig1,axs1 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
table=pd.crosstab(df.policy_csl, df.fraud_reported)
ax=table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,ax=axs1[0])
axs1[0].set_title('Stacked Bar Chart of policy_csl vs Fraud', fontsize=12)
ax.set_xlabel('policy_csl')
ax.set_ylabel('Fraud reported')


df['csl_per_person'] = df.policy_csl.str.split('/', expand=True)[0]
df['csl_per_accident'] = df.policy_csl.str.split('/', expand=True)[1]
print(df['csl_per_person'].head())
print(df['csl_per_accident'].head())
print(df.auto_year.value_counts())

# Deriving the age of the vehicle based on the year value

df['vehicle_age'] = 2023 - df['auto_year']
print(df['vehicle_age'].head(10))

# Factorize according to the time period of the day.
bins = [-1, 3, 6, 9, 12, 17, 20, 24]
names = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
df['incident_period_of_day'] = pd.cut(df.incident_hour_of_the_day, bins, labels=names).astype(object)
print(df[['incident_hour_of_the_day', 'incident_period_of_day']].head(20))

#corr_matrix = df.corr()
#plt.figure(figsize=(10,20))
# Visualize the correlation matrix using a heatmap
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# checking categorcial columns

print(df.select_dtypes(include=['object']).columns)

df = df.drop(columns = [
    'policy_number',
    'policy_csl',
    'insured_zip',
    'policy_bind_date',
    'incident_date',
    'incident_location',
    'auto_year',
    'incident_hour_of_the_day'])

print(df.head(2))

# identify variables with '?' values
unknowns = {}
for i in list(df.columns):
    if (df[i]).dtype == object:
        j = np.sum(df[i] == "?")
        unknowns[i] = j
unknowns = pd.DataFrame.from_dict(unknowns, orient = 'index')
print(unknowns)

print(df.collision_type.value_counts())

ax= df.groupby('collision_type').police_report_available.count().plot.bar(ylim=0,ax=axs1[1])
ax.set_ylabel('Police report')
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
axs1[1].set_title('Collision type according to Police Report',fontsize='12')
plt.subplots_adjust(hspace=0.5)

# Figure 2
print(df.property_damage.value_counts())
fig2,axs2 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
ax= df.groupby('property_damage').police_report_available.count().plot.bar(ylim=0,ax=axs2[0])
ax.set_ylabel('Police report')
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
axs2[0].set_title('Property Damaged Report',fontsize=12)

print(df.police_report_available.value_counts())
plt.subplots_adjust(hspace=0.5)

print(df.columns)

print(df._get_numeric_data().head())
# Checking numeric columns
print(df._get_numeric_data().columns)
print(df.select_dtypes(include=['object']).columns)

print(df.fraud_reported.value_counts())

# Convert non-numeric categorical columns to numeric using LabelEncoder
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Apply one-hot encoding to all categorical columns
df = pd.get_dummies(df)
for col in df.columns:
    unique_values = df[col].unique()
    print(f"Column '{col}': {unique_values}")
X = df.drop(columns = ['fraud_reported'])  #dropping fraud_reported column

y=df['fraud_reported']
# Check for non-numeric values in each column
for col in df.columns:
    if df[col].dtype == object:  # Check if the column contains non-numeric values
        non_numeric_values = df[col][~df[col].str.isnumeric()]
        if len(non_numeric_values) > 0:
            print("______")
            print(f"Non-numeric values found in column '{col}': {non_numeric_values}")

# Print the updated DataFrame
pd.set_option('display.max_columns', None)
print(df.head())
print(X.columns) #showing columns
print(X.count())
plt.show()
