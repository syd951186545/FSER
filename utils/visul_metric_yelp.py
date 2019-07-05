from numpy.random import beta
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.figure(figsize=(8, 4.95))
plt.figure(figsize=(10, 6.18))
plt.style.use('bmh')
xname = [" ", "PMF", "funkSVD", "SVD++", "MTER", "EFM", "DeepCoNN", "FSER"]
# plt.xticks(range(7), xname, rotation=1)
x_mae = [x - 0.20 for x in range(1, 8)]
x_rmse = [x + 0.20 for x in range(1, 8)]
y_mae = [0.7672, 0.7624, 0.7716, 1.1913, 0.8346, 0.8141, 0.7795]
y_rmse = [0.9830, 0.9761, 0.9827, 1.4854, 1.0866, 1.0405, 0.9916]

host1 = host_subplot(212)
par1 = host1.twinx()
host1.set_xlabel("Yelp数据集", fontsize=16)
# host1.set_ylabel("MAE")
# par1.set_ylabel("RMSE")
host1.stem(x_mae, y_mae, linefmt='dodgerblue', markerfmt='_', bottom=0.84)
par1.stem(x_rmse, y_rmse, linefmt='orangered', markerfmt='_', bottom=1.10)
host1.yaxis.get_label().set_color('dodgerblue')
par1.yaxis.get_label().set_color('orangered')
host1.set_xticklabels(xname, fontsize=14)
host1.set_ylim((0.72, 0.84))
par1.set_ylim((0.96, 1.10))
for x, y in zip(x_mae, y_mae):
    host1.text(x, y - 0.01, y, ha='center', va='bottom', fontsize=10)
for x, y in zip(x_rmse[:3], y_rmse[:3]):
    par1.text(x, y - 0.01, y, ha='center', va='bottom', fontsize=10)
for x, y in zip(x_rmse[4:], y_rmse[4:]):
    par1.text(x, y - 0.01, y, ha='center', va='bottom', fontsize=10)

host2 = host_subplot(211)
par2 = host2.twinx()
# host1.set_xlabel("Yelp")
host2.set_ylabel("MAE", fontsize=18)
par2.set_ylabel("RMSE", fontsize=18)
host2.stem(x_mae, y_mae, label="MAE", linefmt='dodgerblue', markerfmt='_')
par2.stem(x_rmse, y_rmse, label="RMSE", linefmt='orangered', markerfmt='_')
host2.yaxis.get_label().set_color('dodgerblue')
par2.yaxis.get_label().set_color('orangered')
host2.set_ylim((1.18, 1.24))
par2.set_ylim((1.46, 1.50))
host2.set_xticklabels(xname, fontsize=14)
for x, y in zip(x_mae, y_mae):
    host2.text(x, y + 0.001, y, ha='center', va='bottom', fontsize=10)

par2.text(4.2, 1.4864, 1.4854, ha='center', va='bottom', fontsize=10)
leg = plt.legend(fontsize=16)
leg.texts[0].set_color('dodgerblue')
leg.texts[1].set_color('orangered')

# plt.subplot(211)
# x = range(7)
# y_2 = [0.7672, 0.7624, 0.7716, 1.1913, 0.8141, 0.8, 0.7795]
# plt.ylim((0.83, 1.2))
# plt.xticks(x, xname, rotation=1)
# plt.stem(x, y_2, linefmt='grey', markerfmt='D')
plt.savefig("../Result/yelp_MAE_RMSE")
plt.show()
