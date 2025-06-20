import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Load your data
# df = pd.read_excel('your_file.xlsx')  # if loading from Excel
df = pd.read_excel('labels_sex_age copy.xlsx')

# Plot 1: Age distribution by gender
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='gender',bins=60, kde=True, palette='Set1', element='step')
# plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Gender', labels=['Male', 'Male distribution', 'Female', 'Female distribution' ])  # adjust if needed
plt.grid(True)
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=300)
plt.savefig("age_distribution.eps", format='eps')
plt.show()

# Plot 2: Count of samples by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df, palette='Set1')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks([1, 0], ['Male', 'Female'])
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("gender_distribution.png", dpi=300)
plt.savefig("gender_distribution.eps", format='eps')
plt.show()
