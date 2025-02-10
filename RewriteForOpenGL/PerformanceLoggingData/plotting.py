import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
csv_file = 'RewriteForOpenGL\PerformanceLoggingData\\Setup1Run1.csv'  # Replace with your file path
data = pd.read_csv(csv_file)

# Select two columns for plotting
x_column = 'frameIndex'  # Replace with the name of your desired X-axis column
y_column = 'chargeDrawing'

fps = "FPS"  # Replace with the name of your desired Y-axis column

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(data[x_column], data[y_column]/(data[fps]*100), marker='o', linestyle='-', label=f'{y_column} vs {x_column}', s=5)

# Customize the plot
plt.title(f'{y_column} vs {x_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.legend()
plt.grid(True)
#plt.xscale("log")

# Show the plot
plt.show()
