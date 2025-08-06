import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

multi_num_UE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
num_point = 9

hun = np.array([0.0342, 0.0334, 0.0382, 0.0447, 0.0401, 0.0405, 0.0414, 0.0636, 0.0993]) / 20
pso = np.array([80.7172, 248.1016, 486.1518, 828.3169, 1.0800e+03, 1.6677e+03, 1.9062e+03, 2.5370e+03, 2.8415e+03]) / 20
ga = np.array([144.8361, 262.1913, 411.0212, 653.3375, 883.0785, 1.1763e+03, 1.5535e+03, 1.9061e+03, 2.3573e+03]) / 20

plt.plot(hun, label='HUN', marker='D', color='#c82423', markersize=5)
plt.plot(pso, label='PSO', marker='D', color='#3480b8', markersize=5)
plt.plot(ga, label='GA', marker='D', color='#FFC000', markersize=5)
xtick = [a*3 for a in multi_num_UE[:num_point]]
plt.xticks([a for a in range(num_point)], xtick)
plt.xlabel('UE number', fontsize=14)
plt.ylabel('Time (s)', fontsize=14)
plt.grid()
plt.legend()
plt.show()
print(hun)