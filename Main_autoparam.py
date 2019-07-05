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
    , uia_senti_dic_test = dataprocess(U_num, I_num, F_num, W_num)

for param_search in range(1000):
    # -------Hyper parameter----------
    print("----------" + str(param_search) + "--------------------")
    param_file = open("./param_def_lmd.txt", "a")
    U_dim = 24
    I_dim = 12
    F_dim = 12
    W_dim = 24
    # num_iter = random.choice([i * 2000 for i in range(1, 11)])
    num_iter = 6000
    # minibatch = random.choice([200, 400, 700, 1000])
    minibatch = 200
    lmd_reg = 10 * round(random.random(), 2)
    lmd_r = 1
    lmd_s = round(random.random(), 2)
    lmd_o = round(random.random(), 2)
    lmd_bpr = 0
    # neg_sample_rate = random.choice([0.01, 0.1, 0.2, 0.5])  # 这个负采样为选择负样本的概率
    neg_sample_rate = 0
    lr = 0.1
    param_s = (num_iter, minibatch, lmd_r, lmd_s, lmd_o, lmd_bpr, lmd_reg, lr, neg_sample_rate)
    # --------------------------
    print(U_dim, I_dim, F_dim, W_dim)
    print('Start Training ...')
    start_time = time.time()
    U, I, F, W = minibatch_adagradSGD_train(uiaw_list, uw_frequency_mat,
                                            ui_rating_dic, uia_senti_dic, iaw_frequency_dic,
                                            U_dim, I_dim, F_dim, W_dim, U_num, I_num, F_num + 1, W_num,
                                            num_iter, lmd_reg, lmd_r, lmd_s, lmd_o, neg_sample_rate, lmd_bpr,
                                            minibatch, lr, ui_rating_dic_test, uia_senti_dic_test, random_seed=0,
                                            eps=1e-8)
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
    param_file.write("lmd_r={}\t,lmd_s={}\t,lmd_o={}\t,lmd_reg={}".format(lmd_r, lmd_s, lmd_o, lmd_reg))
    param_file.write("\n")
    param_file.write(str(nowTime) + " \tRMSE:" + str(RMSEv) + " \tMAE:" + str(MAEv))
    param_file.write("\n")
    param_file.close()

    # ---------Output to file  or Not------------
    # Output(nowTime, U, I, F, W, U_num, iaw_frequency_dic, word_dic, aspect_dic, rec_item, ui_rating_dic_test,
    #        iaw_frequency_test_dic)
