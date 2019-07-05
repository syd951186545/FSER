from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.plot(x, y, 'ro-')
# plt.plot(x, y, 'bo-')
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)  # 限定纵轴的范围

x = range(0, 111, 5)
y_yelp = [0, 0.28, 0.49, 0.62, 0.715, 0.83, 0.88, 0.91, 0.935, 0.945, 0.95, 0.953, 0.956, 0.96, 0.964, 0.966, 0.969,
          0.971, 0.973, 0.975, 0.979, 0.98, 0.982]
y_dianp = [0, 0.31, 0.45, 0.56, 0.58, 0.65, 0.68, 0.7, 0.76, 0.785, 0.82, 0.845, 0.85, 0.875, 0.89, 0.91, 0.915, 0.92,
           0.94, 0.94, 0.95, 0.95, 0.955
           ]
plt.plot(x, y_yelp, mec='r', mfc='w', label='YELP')
plt.plot(x, y_dianp, ms=10, label='大众点评')
plt.legend(fontsize=14)  # 让图例生效

plt.xticks(range(0, 111, 10), rotation=1)
plt.yticks([i / 10 for i in range(11)], rotation=1)
plt.ylim((0, 1))
plt.margins(0)
plt.xlabel('出现频次最多的前K属性', fontsize=12)  # X轴标签
plt.ylabel("占总属性出现频次比例", fontsize=12)  # Y轴标签
# plt.yticks([0.750, 0.800, 0.850])
# plt.title("A simple plot")  # 标题
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.savefig('../Result/word_freq.jpg', dpi=900)
plt.show()