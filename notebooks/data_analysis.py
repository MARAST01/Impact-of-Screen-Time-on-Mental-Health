import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the data
df = pd.read_csv('../data/digital_diet_mental_health.csv')

# Basic statistics
print("\n=== Basic Statistics ===")
print("\nNumerical Columns Summary:")
print(df.describe())

print("\nCategorical Columns Summary:")
print(df[['gender', 'location_type']].value_counts())

# Correlation analysis
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('../data/correlation_matrix.png')
plt.close()

# Screen time analysis
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='daily_screen_time_hours', bins=30)
plt.title('Distribution of Daily Screen Time')
plt.xlabel('Hours per Day')
plt.savefig('../data/screen_time_distribution.png')
plt.close()

# Mental health score analysis
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='mental_health_score', bins=30)
plt.title('Distribution of Mental Health Scores')
plt.xlabel('Score')
plt.savefig('../data/mental_health_distribution.png')
plt.close()

# Screen time vs Mental health
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='daily_screen_time_hours', y='mental_health_score')
plt.title('Screen Time vs Mental Health Score')
plt.xlabel('Daily Screen Time (hours)')
plt.ylabel('Mental Health Score')
plt.savefig('../data/screen_time_vs_mental_health.png')
plt.close()

# Age groups analysis
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], 
                        labels=['0-18', '19-30', '31-45', '46-60', '60+'])

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='age_group', y='mental_health_score')
plt.title('Mental Health Score by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mental Health Score')
plt.savefig('../data/mental_health_by_age.png')
plt.close()

# Gender analysis
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='gender', y='mental_health_score')
plt.title('Mental Health Score by Gender')
plt.xlabel('Gender')
plt.ylabel('Mental Health Score')
plt.savefig('../data/mental_health_by_gender.png')
plt.close()

# Location type analysis
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='location_type', y='mental_health_score')
plt.title('Mental Health Score by Location Type')
plt.xlabel('Location Type')
plt.ylabel('Mental Health Score')
plt.savefig('../data/mental_health_by_location.png')
plt.close()

# Additional statistics
print("\n=== Additional Statistics ===")
print("\nAverage Screen Time by Age Group:")
print(df.groupby('age_group')['daily_screen_time_hours'].mean())

print("\nAverage Mental Health Score by Age Group:")
print(df.groupby('age_group')['mental_health_score'].mean())

print("\nCorrelation between Screen Time and Mental Health Score:")
correlation = df['daily_screen_time_hours'].corr(df['mental_health_score'])
print(f"Correlation coefficient: {correlation:.3f}")

# Save summary statistics to a text file
with open('../data/analysis_summary.txt', 'w') as f:
    f.write("=== Analysis Summary ===\n\n")
    f.write("Basic Statistics:\n")
    f.write(df.describe().to_string())
    f.write("\n\nCorrelation between Screen Time and Mental Health Score:\n")
    f.write(f"Correlation coefficient: {correlation:.3f}\n")
    f.write("\nAverage Screen Time by Age Group:\n")
    f.write(df.groupby('age_group')['daily_screen_time_hours'].mean().to_string())
    f.write("\n\nAverage Mental Health Score by Age Group:\n")
    f.write(df.groupby('age_group')['mental_health_score'].mean().to_string()) 