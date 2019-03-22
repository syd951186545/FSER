import datetime
import random
import time
import numpy as np
from Core.Train import minibatch_adagradSGD_train
from Core.DataLoader import dataprocess
from Core.Output import Output

U_num = 10719
I_num = 10410
F_num = 104
W_num = 1019

uiaw_list, uw_frequency_mat, ui_rating_dic, uia_senti_dic, iaw_frequency_dic, ui_rating_dic_test, word_dic, aspect_dic,iaw_frequency_test_dic \
    = dataprocess(U_num, I_num, F_num, W_num)

# -------Hyper parameter----------



# print("----------" + str(param_search) + "--------------------")
param_file = open("./param.txt", "a")
U_dim = 48
I_dim = 24
F_dim = 24
W_dim = 48
num_iter = 3000
minibatch = 400
lmd_reg = 0.1
lmd_r = 1
lmd_s = 0.1
lmd_o = 0.3
lmd_bpr = 0.8
neg_sample_rate = random.choice([0.1, 0.2, 0.5])  # 这个负采样为选择负样本的概率
lr = random.choice([0.01, 0.1, 0.2])
param_s = (num_iter, minibatch, lmd_reg, lmd_r, lmd_s, lmd_o, lmd_bpr, lr, neg_sample_rate)
# --------------------------
print(U_dim, I_dim, F_dim, W_dim)
print('Start Training ...')
start_time = time.time()
U, I, F, W = minibatch_adagradSGD_train(uiaw_list, uw_frequency_mat,
                                        ui_rating_dic, uia_senti_dic, iaw_frequency_dic,
                                        U_dim, I_dim, F_dim, W_dim, U_num, I_num, F_num + 1, W_num,
                                        num_iter, lmd_reg, lmd_r, lmd_s, lmd_o, neg_sample_rate, lmd_bpr,
                                        minibatch, lr, random_seed=0, eps=1e-8)
train_time = time.time() - start_time

# ---------Evaluate  or Not------------
print('Evaluate...')
evaluate_res = []
rec_item = np.einsum('ma,na ->mn ', U, np.hstack((I, np.tile(F[104], (I_num, 1)))))
for key in ui_rating_dic_test.keys():
    real_rating = ui_rating_dic_test[key]
    key = key[1:-1].split(",")
    u_id = int(key[0])
    i_id = int(key[1])
    rec_rating = rec_item[u_id][i_id]
    evaluate_res.append([u_id, i_id, real_rating, rec_rating])
from FSER.Metric import metric

metric = metric.Metric()
print("MAE:")
MAEv = metric.MAE(evaluate_res)
print(MAEv)
print("RMSE:")
RMSEv = metric.RMSE(evaluate_res)
print(RMSEv)
print(train_time)

nowTime = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
param_file.write(str(param_s))
param_file.write("\n")
param_file.write(str(nowTime) + " RMSE:" + str(RMSEv) + " MAE:" + str(MAEv))
param_file.write("\n")
param_file.close()

# ---------Output to file  or Not------------
Output(nowTime, U, I, F, W, U_num, iaw_frequency_dic, word_dic, aspect_dic, rec_item, ui_rating_dic_test,iaw_frequency_test_dic)
