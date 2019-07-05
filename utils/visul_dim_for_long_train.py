from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pathdir = "../Result/FSER_/"

f24 = pathdir + "train_every100_10000_24"
f48 = pathdir + "train_every100_10000_48"
f64 = pathdir + "train_every100_10000_64"
f128 = pathdir + "train_every100_10000_128"
f192 = pathdir + "train_every100_10000_192"
files = [f24, f48, f64, f128, f192]
Yrmse_dic = {}
Ymae_dic = {}
train_time_dic = {}
for file in files:
    with open(file, "r") as f:
        i = 0
        s = ""
        for line in f.readlines():
            if i == 0:
                s = line.replace("\n", "")
                Yrmse_dic[s] = []
                Ymae_dic[s] = []
                train_time_dic[s] = []
                i += 1
            else:
                if i % 3 == 1:
                    line = line.split("\t")[0]
                    mae = round(float(line.split(":")[1]), 4)
                    Ymae_dic[s].append(mae)
                if i % 3 == 2:
                    line = line.split("\t")[0]
                    rmse = round(float(line.split(":")[1]), 4)
                    Yrmse_dic[s].append(rmse)
                if i % 3 == 0:
                    mae = round(float(line.split(":")[1]), 4)
                    train_time_dic[s].append(mae)
                i += 1
plt.figure(figsize=(16, 9))

plt.subplot(221)
for key in Yrmse_dic.keys():
    # key_s = key.split("-")
    # line_label = "dim I,F={}/dim U,W={}".format(key_s[0], key_s[1])
    x = range(100, 4000, 100)
    plt.xlim((100, 3800))
    plt.plot(x[0:39], Yrmse_dic[key][1:40], mec='r', mfc='w')
    plt.xlabel('迭代次数', fontsize=16)  # X轴标签
    plt.ylabel("RMSE值", fontsize=16)  # Y轴标签
    x_names = range(100, 4000, 1000)
    plt.xticks(x_names, rotation=1)

plt.subplot(223)
for key in Ymae_dic.keys():
    # key_s = key.split("-")
    # line_label = "dim I,F={}/dim U,W={}".format(key_s[0], key_s[1])
    x = range(100, 4000, 100)
    plt.xlim((100, 3800))
    plt.plot(x[0:39], Ymae_dic[key][1:40], mec='r', mfc='w')
    plt.xlabel('迭代次数', fontsize=16)  # X轴标签
    plt.ylabel("MAE值", fontsize=16)  # Y轴标签
    x_names = range(100, 4000, 1000)
    plt.xticks(x_names, rotation=1)


plt.subplot(224)
for key in Ymae_dic.keys():
    # key_s = key.split("-")
    # line_label = "dim I,F={}/dim U,W={}".format(key_s[0], key_s[1])
    x = range(0, 10000, 100)
    plt.xlim((1000, 9900))
    plt.plot(x[10:], Ymae_dic[key][10:], mec='r', mfc='w')
    plt.xlabel('迭代次数', fontsize=16)  # X轴标签
    plt.ylabel("MAE值", fontsize=16)  # Y轴标签
    x_names = range(1000, 10000, 2000)
    plt.xticks(x_names, rotation=1)

plt.subplot(222)
for key in Yrmse_dic.keys():

    key_s = key.split("-")
    line_label = "dim I,F={}/dim U,W={}".format(key_s[0], key_s[1])
    x = range(0, 10000, 100)
    plt.xlim((1000, 9900))
    plt.plot(x[10:], Yrmse_dic[key][10:], mec='r', mfc='w', label=line_label)
    plt.xlabel('迭代次数', fontsize=16)  # X轴标签
    plt.ylabel("RMSE值", fontsize=16)  # Y轴标签
    x_names = range(1000, 10000, 2000)
    plt.xticks(x_names, rotation=1)

plt.legend(fontsize=16)  # 让图例生效


plt.margins(0)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.title("A simple plot")  # 标题
plt.savefig('../Result/4he1_3.jpg', dpi=1600)
plt.show()
