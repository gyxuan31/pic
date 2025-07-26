import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 10, 100)
y = np.concatenate([np.linspace(0, 10, 50), np.linspace(50, 80, 50)])

# 创建两个子图，共享x轴
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5), height_ratios=[1, 1])

# 设置第一个子图：y轴 50-80
ax1.plot(x, y)
ax1.set_ylim(50, 80)

# 设置第二个子图：y轴 0-10
ax2.plot(x, y)
ax2.set_ylim(0, 10)

# 去掉断层中间的spines（轴线）
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 添加“断轴”的小斜线标记
d = .015  # 斜线的长度
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right

# 标签设置
ax2.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax2.set_ylabel('Y Axis')

plt.tight_layout()
plt.show()
