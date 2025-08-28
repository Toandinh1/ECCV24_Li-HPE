import matplotlib.pyplot as plt
import numpy as np

# Example data
x_labels = [r'PCK$_{50}$', r'PCK$_{40}$', r'PCK$_{30}$', r'PCK$_{20}$']
x = np.arange(len(x_labels))  # Numerical positions for the bars

# Y-values for the six different methods
y_values = [
    [79.96, 71.70, 60.09, 41.65],
    [78.20, 68.89, 55.64, 35.45],
    [77.23, 67.44, 53.78, 33.95],
    [76.69, 66.82, 52.93, 32.45],
    [67.23, 55.95, 41.24, 23.32],
    [63.40, 54.48, 40.70, 23.40]
]


labels = ['HPE + Stacked AE', 'HPE + Traditional AE', 'HPE + Mean Filter', 'HPE + Gaussian Filter', 'HPE', 'Basic CNN']
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']  # Different colors for each method
hatch_patterns = ['/', '\\', 'x', '+', 'o', '|']  # Changed '-' to '|' for vertical lines

# Set up the plot
plt.style.use('seaborn-whitegrid')  # White background with grid
params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"]
}
plt.rcParams.update(params)

plt.figure(figsize=(10, 7), dpi=150)

bar_width = 0.15  # Width of bars

# Plot the bars for each method with hatching patterns and custom colors
for i, (y, pattern, color) in enumerate(zip(y_values, hatch_patterns, colors)):
    plt.bar(x + i * bar_width, y, width=bar_width, label=labels[i], hatch=pattern, color=color, linewidth=2)

# Configure the axes and labels
plt.ylabel('Percentage (\%)', fontsize=28)
plt.xticks(x + bar_width * 2.5, x_labels, fontsize=24)
plt.yticks(fontsize=24)

# Add legend with box
plt.legend(loc='upper right', fontsize=15, frameon=True, facecolor='white', edgecolor='black')  # Add box for legend

# Enable grid with black lines
plt.grid(visible=True, color='black', linestyle='--', alpha=0.7)  # Set grid lines to black

# Save and show the plot
plt.savefig('AWGN_bar.pdf', format='pdf', bbox_inches='tight', dpi = 500)
plt.show()
