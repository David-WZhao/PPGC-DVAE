import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
# 设置字体大小
plt.rcParams.update({
    'font.size': 40,         # 设置全局字体大小
    'axes.titlesize': 40,    # 设置标题字体大小
    'axes.labelsize': 35,    # 设置轴标签字体大小
    'xtick.labelsize': 35,   # 设置x轴刻度标签字体大小
    'ytick.labelsize': 35    # 设置y轴刻度标签字体大小
})
# Example data points for two curves
x1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
y1 = np.array([0.0, 25, 80, 150, 320, 400, 200, 50, 10, 0])

x2 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
y2 = np.array([0.0, 50, 100, 150, 280, 300, 200, 100, 50, 0])

# Pchip interpolation to create smooth curves
pchip1 = PchipInterpolator(x1, y1)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = pchip1(x1_smooth)

pchip2 = PchipInterpolator(x2, y2)
x2_smooth = np.linspace(x2.min(), x2.max(), 300)
y2_smooth = pchip2(x2_smooth)

# Plot the smooth curves
plt.figure(figsize=(10, 8))
plt.plot(x1_smooth, y1_smooth, linestyle='-', color='skyblue')
plt.plot(x2_smooth, y2_smooth, linestyle='-', color='orange')
# 添加竖直虚线
plt.axvline(x=0.4, color='grey', linestyle='--')
# Set the labels and title
plt.xlabel('Normalized Value Range')
plt.ylabel('Sample Count')
plt.title('Distribution of Alpha Helix')


# Show the grid
plt.grid(True)


plt.show()