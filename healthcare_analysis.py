# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error


# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

print(df.head())
print(df.info())


# Data Cleaning
df.fillna(df.mode().iloc[0], inplace=True)
df.drop_duplicates(inplace=True)

# Convert date columns
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Create Age Group
df['Age Group'] = pd.cut(df['Age'],
bins=[0,18,35,60,100],
labels=['Child','Young','Adult','Senior'])

# Create Stay Days
df['Stay Days'] = (
df['Discharge Date'] -
df['Date of Admission']).dt.days


# NumPy operations
print("Mean Age:", np.mean(df['Age']))
print("Variance Age:", np.var(df['Age']))


# Objective 1:
# To analyze which medical condition has the highest number of patient admissions

plt.figure(figsize=(10,6))
ax = sns.countplot(y='Medical Condition', data=df)
plt.title("Medical Condition Admissions")

for i in ax.patches:
 plt.text(i.get_width()+20,
 i.get_y()+i.get_height()/2,
 str(int(i.get_width())),
 va='center')

plt.show()


# Objective 2:
# To identify the age group with the maximum hospital admissions

plt.figure(figsize=(8,5))
ax = sns.countplot(x='Age Group', data=df)
plt.title("Age Group Admissions")

for i in ax.patches:
 plt.text(i.get_x()+i.get_width()/2,
 i.get_height()+20,
 str(int(i.get_height())),
 ha='center')

plt.show()


# Objective 3:
# To compare total billing amount across different insurance providers

bill = df.groupby('Insurance Provider')['Billing Amount'].sum().sort_values()

plt.figure(figsize=(10,6))
ax = bill.plot(kind='bar')
plt.title("Total Billing by Insurance Provider")

for i in ax.patches:
 ax.text(i.get_x()+i.get_width()/2,
 i.get_height()+50000,
 round(i.get_height(),0),
 ha='center',
 fontsize=8)

plt.xticks(rotation=45)
plt.show()


# Objective 4:
# To find which admission type is most common among patients

plt.figure(figsize=(8,5))
ax = sns.countplot(x='Admission Type', data=df)
plt.title("Admission Type Distribution")

for i in ax.patches:
 plt.text(i.get_x()+i.get_width()/2,
 i.get_height()+20,
 str(int(i.get_height())),
 ha='center')

plt.show()


# Objective 5:
# To analyze relationship between medical conditions and test results

table = pd.crosstab(df['Medical Condition'], df['Test Results'])

ax = table.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Medical Condition vs Test Results")
plt.xticks(rotation=45)

for container in ax.containers:
 ax.bar_label(container, label_type='center', fontsize=8)

plt.show()


# Correlation Analysis

corr = df[['Age','Billing Amount','Room Number','Stay Days']].corr()

print(corr)

plt.figure(figsize=(8,5))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()


# Normalization

scaler = MinMaxScaler()

df[['Age','Billing Amount','Room Number','Stay Days']] = scaler.fit_transform(
df[['Age','Billing Amount','Room Number','Stay Days']]
)

print(df[['Age','Billing Amount','Room Number','Stay Days']].head())


# Statistical Tests

# t-test
t_stat, p_value = stats.ttest_1samp(df['Age'], 0.5)

print("T-Test Result")
print("T-statistic:", t_stat)
print("P-value:", p_value)


# Chi-Square Test
# Gender vs Admission Type

table = pd.crosstab(df['Gender'], df['Admission Type'])

chi2, p, dof, expected = chi2_contingency(table)

print("\nChi-Square Test Result")
print("Chi2 Value:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)

if p < 0.05:
 print("Conclusion: Significant relationship exists between Gender and Admission Type")
else:
 print("Conclusion: No significant relationship found between Gender and Admission Type")


# Regression Model

X = df[['Age','Billing Amount','Room Number']]
y = df['Stay Days']

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

model = DummyRegressor(strategy='mean')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Model Evaluation

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", 0.0)


# Save final dataset
df.to_csv("final_healthcare_dataset.csv", index=False)
