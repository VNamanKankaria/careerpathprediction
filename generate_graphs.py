import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the models
models = ["Pure ML", "Pure CBR", "Hybrid (CBR + ML)"]

# Define performance scores for different criteria
accuracy = [80, 70, 90]  # Hypothetical Accuracy %
adaptability = [60, 80, 95]  # How well the model adapts (0-100)
learning_efficiency = [90, 70, 85]  # Lower for CBR since it needs more cases
handling_unseen_cases = [50, 90, 95]  # Pure ML struggles with unseen cases

# X-axis categories
categories = ["Accuracy", "Adaptability", "Learning Efficiency", "Handling Unseen Cases"]

# Data for plotting
data = np.array([accuracy, adaptability, learning_efficiency, handling_unseen_cases])

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x_indexes = np.arange(len(categories))

# Plot bars for each model
for i, model in enumerate(models):
    ax.bar(x_indexes + i * bar_width, data[:, i], width=bar_width, label=model)

# Formatting
ax.set_xlabel("Evaluation Metrics")
ax.set_ylabel("Score (0-100)")
ax.set_title("Comparison of Pure ML, Pure CBR, and Hybrid AI Model")
ax.set_xticks(x_indexes + bar_width)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.show()
