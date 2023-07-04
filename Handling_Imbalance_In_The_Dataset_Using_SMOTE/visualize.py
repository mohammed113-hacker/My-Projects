
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('sample-data/insurance_claims.csv')
print(df.head(10))
print(df.dtypes)
print(df.columns)
print(df.shape)
print(df.nunique())
plt.style.use('tableau-colorblind10')

# Figure 1
fig1, axs1 = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
ax = sns.countplot(x='fraud_reported', data=df,hue='fraud_reported', ax=axs1[0])
axs1[0].set_title('Count of fraud reported')
print(df['fraud_reported'].value_counts())
print(df['incident_state'].value_counts())
ax = df.groupby('incident_state').fraud_reported.count().plot(kind='bar', ax=axs1[1])
ax.set_ylabel('Fraud reported')
axs1[1].set_title('Count of Fraud Reported by Incident State')
plt.subplots_adjust(hspace=0.5)

# Figure 2
fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
table=pd.crosstab(df.age, df.fraud_reported)
print(table)
ax=table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,ax=axs2[0])
axs2[0].set_title('Stacked Bar Chart of Age vs Fraud Reported', fontsize=12)
ax.set_xlabel('Age')
ax.set_ylabel('Fraud reported')
ax = df.groupby('incident_date').total_claim_amount.count().plot.bar(ylim=0,ax=axs2[1])
axs2[1].set_title('incident by date')
ax.set_ylabel('Claim amount ($)')
plt.subplots_adjust(hspace=0.5)

# Figure 3
fig3,axs3 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
ax = df.groupby('policy_state').fraud_reported.count().plot.bar(ylim=0,ax=axs3[0])
axs3[0].set_title('Fraud reported in policy state')
ax.set_ylabel('Fraud reported')
plt.rcParams['figure.figsize'] = [10, 6]
table=pd.crosstab(df.policy_state, df.fraud_reported)
print(table)
ax = table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,ax=axs3[1])
axs3[1].set_title('Stacked Bar Chart of Policy State vs Fraud Reported', fontsize=12)
ax.set_xlabel('Policy state')
ax.set_ylabel('Fraud reported')
plt.subplots_adjust(hspace=0.5)

# Figure 4
fig4,axs4 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
ax = df.groupby('incident_type').fraud_reported.count().plot.bar(ylim=0,ax=axs4[0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="right")
ax.set_ylabel('Fraud reported')
ax = sns.countplot(x='incident_state', data=df,ax=axs4[1])
axs4[1].set_title('count on the incident state')
plt.subplots_adjust(hspace=0.5)

# Figure 5
fig5,axs5 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
ax = sns.countplot(y='insured_education_level', data=df,ax=axs5[0],saturation=0.80)
axs5[0].set_title('Number of insured_education_level people')
table=pd.crosstab(df.insured_education_level, df.fraud_reported)
print(table)
ax=table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,ax=axs5[1])
axs5[1].set_title('Stacked Bar Chart of insured education vs Fraud reported', fontsize=12)
ax.set_xlabel('Insured_education_level')
ax.set_ylabel('Fraud reported')
plt.subplots_adjust(hspace=0.5)

#Figure 6
fig6 = plt.figure(figsize=(10,6))
ax = (df['insured_sex'].value_counts()*100.0 /len(df)).plot.pie(autopct='%.1f%%', labels=['Male','Female'])
ax.set_title('% Gender')

#Figure 7
fig7,axs7 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
table=pd.crosstab(df.insured_sex, df.fraud_reported)
print(table)
ax=table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,ax=axs7[0])
axs7[0].set_title('Stacked Bar Chart of insured_sex vs Fraud', fontsize=12)
ax.set_xlabel('Insured_sex')
ax.set_ylabel('Fraud reported')
plt.subplots_adjust(hspace=0.5)

#Figure 8
fig = plt.figure(figsize=(10,6))
ax = (df['incident_severity'].value_counts()*100.0 /len(df)).plot.pie(autopct='%.1f%%', labels = ['Major Damage', 'Total Loss', 'Minor Damage', 'Trivial Damage'],fontsize=12)

#Figure 9
fig9,axs9 = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
ax = sns.countplot(data=df,ax=axs9[0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
axs9[0].set_title('Insured Hobbies')
ax= df.groupby('auto_make').vehicle_claim.count().plot.bar(ylim=0,ax=axs9[1])
ax.set_ylabel('Vehicle claim')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.subplots_adjust(hspace=0.5)

print(df["insured_occupation"].value_counts())

plt.show()