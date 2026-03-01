import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the results from the CSV we created in the last step
try:
    df = pd.read_csv('eval_results.csv')
except FileNotFoundError:
    print("Error: 'eval_results.csv' not found. Run your batch_eval.py first!")
    exit()

# 2. Group the data by Score and count them
# Example: How many 5s did we get? How many 1s?
score_counts = df['Score'].value_counts().sort_index()

# 3. Create the Plot
plt.figure(figsize=(10, 6))
score_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Add Labels and Title
plt.title('AI Performance: Score Distribution', fontsize=16)
plt.xlabel('Score (1 = Poor, 5 = Excellent)', fontsize=12)
plt.ylabel('Number of Test Cases', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Save the chart as an image
plt.savefig('eval_chart.png')
print("✅ Success! Your chart has been saved as 'eval_chart.png'")