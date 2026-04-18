# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Objective 1:
# To analyze how weather conditions affect air pollution

# Load dataset
df = pd.read_excel("air_pollution_weather_dataset.xlsx")

print(df.head())
print(df.info())


# ================= UNIT II =================
# Data Cleaning and Manipulation

# Handling missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Removing duplicates
df.drop_duplicates(inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# NumPy usage (explicit)
print("Mean PM2.5:", np.mean(df['PM2.5']))
print("Variance PM2.5:", np.var(df['PM2.5']))


# ================= UNIT III =================
# Data Visualization

# Wind vs Pollution
plt.figure()
sns.scatterplot(x='Wind Speed', y='PM2.5', data=df)
plt.title("Wind Speed vs Pollution")
plt.show()

# Temperature vs Pollution
plt.figure()
sns.scatterplot(x='Temperature', y='PM2.5', data=df)
plt.title("Temperature vs Pollution")
plt.show()

# Humidity vs Pollution
plt.figure()
sns.scatterplot(x='Humidity', y='PM2.5', data=df)
plt.title("Humidity vs Pollution")
plt.show()

# City comparison
plt.figure()
df.groupby('City')['PM2.5'].mean().plot(kind='bar')
plt.title("City-wise Pollution")
plt.show()

# Distribution (IMPORTANT - syllabus)
plt.figure()
sns.histplot(df['PM2.5'], kde=True)
plt.title("PM2.5 Distribution")
plt.show()


# ================= UNIT IV =================
# Exploratory Data Analysis

# Summary statistics
print(df.describe())

# Correlation
corr = df.corr(numeric_only=True)
print(corr)

# Heatmap
plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Outlier detection
plt.figure()
sns.boxplot(df['PM2.5'])
plt.title("Outliers in PM2.5")
plt.show()


# ================= UNIT V =================
# Statistical Analysis

# t-test
t_stat, p_value = stats.ttest_1samp(df['PM2.5'], 100)

print("T-statistic:", t_stat)
print("P-value:", p_value)


# ================= UNIT VI =================
# Machine Learning

# Regression (Supervised Learning)
X = df[['Temperature', 'Humidity', 'Wind Speed']]
y = df['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Save dataset
df.to_excel("final_pollution_dataset.xlsx", index=False)