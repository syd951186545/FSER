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
y_mae = [0.7011, 0.7091, 0.7256, 1.0241, 0.7765, 0.7492, 0.7167]
y_rmse = [0.8963, 0.8928, 0.9131, 1.2776, 1.0005, 0.9536, 0.9011]

host1 = host_subplot(212)
par1 = host1.twinx()
host1.set_xlabel("大众点评数据集",fontsize=16)
# host1.set_ylabel("MAE")
# par1.set_ylabel("RMSE")
host1.stem(x_mae, y_mae, linefmt='dodgerblue', markerfmt='_', bottom=0.80)
par1.stem(x_rmse, y_rmse, linefmt='orangered', markerfmt='_', bottom=0.98)
host1.yaxis.get_label().set_color('dodgerblue')
par1.yaxis.get_label().set_color('orangered')
host1.set_xticklabels(xname, fontsize=14)
host1.set_ylim((0.68, 0.80))
par1.set_ylim((0.88, 0.98))
for x, y in zip(x_mae, y_mae):
    host1.text(x, y - 0.01, y, ha='center', va='bottom', fontsize=12)
for x, y in zip(x_rmse[:3], y_rmse[:3]):
    par1.text(x, y - 0.01, y, ha='center', va='bottom', fontsize=12)
for x, y in zip(x_rmse[5:], y_rmse[5:]):
    par1.text(x, y - 0.01, y, ha='center', va='bottom', fontsize=12)

host2 = host_subplot(211)
par2 = host2.twinx()
# host1.set_xlabel("Yelp")
host2.set_ylabel("MAE", fontsize=18)
par2.set_ylabel("RMSE", fontsize=18)
host2.stem(x_mae, y_mae, label="MAE", linefmt='dodgerblue', markerfmt='_')
par2.stem(x_rmse, y_rmse, label="RMSE", linefmt='orangered', markerfmt='_')
host2.yaxis.get_label().set_color('dodgerblue')
par2.yaxis.get_label().set_color('orangered')
host2.set_ylim((1.0, 1.10))
par2.set_ylim((0.98, 1.35))
host2.set_xticklabels(xname, fontsize=14)
for x, y in zip(x_mae, y_mae):
    host2.text(x, y + 0.001, y, ha='center', va='bottom', fontsize=12)

par2.text(4.2, 1.2790, 1.2776, ha='center', va='bottom', fontsize=12)
par2.text(5.2, 1.0025, 1.0005, ha='center', va='bottom', fontsize=12)

leg = plt.legend(fontsize=16)
leg.texts[0].set_color('dodgerblue')
leg.texts[1].set_color('orangered')

# plt.subplot(211)
# x = range(7)
# y_2 = [0.7672, 0.7624, 0.7716, 1.1913, 0.8141, 0.8, 0.7795]
# plt.ylim((0.83, 1.2))
# plt.xticks(x, xname, rotation=1)
# plt.stem(x, y_2, linefmt='grey', markerfmt='D')
plt.savefig("../Result/dianping_MAE_RMSE")
plt.show()
