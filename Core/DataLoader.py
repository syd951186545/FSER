# Multi-decomposition 2018/09/01 SYD
import _thread
import math
import random
import time
import numpy as np
from sklearn.cluster import KMeans
from Core.config import config


def get_index(key):
    index = key[1:-1].split(',')
    for i in range(3):
        index[i] = int(index[i])
    return index


def dataprocess(U_num=10719, I_num=10410, F_num=104, W_num=1019):
    """
    @Iuput
    ------------------------------------------------------
    *file:uiawr_train(test).entry
    *format:user,item,aspect,word,rating
    *example:
    "
    0,0,0,0,5.0
    0,1,11,8,3.0
    "
    ------------------------------------------------------
    *file:aspect.map
    *format:aspect_id=aspect
    *example:
    "
    0=portion
    1=beef
    2=cheese
    "
    ------------------------------------------------------
    *file:word.senti.map
    *format:word_id=word=senti
    *example:
    "
    0=half=-1
    1=huge=1
    2=little=-1
    "
    ------------------------------------------------------
    """

    # -----Input file-----r:rating/s:sentiment/ u:user/a:aspect/i:item/w:word
    uiaw_train_file = open("./Data/"+config.dataset_name+"/uiawr_id.train", encoding='UTF-8')
    uiaw_test_file = open("./Data/"+config.dataset_name+"/uiawr_id.test", encoding='UTF-8')
    aspect_dic_file = open("./Data/"+config.dataset_name+"/aspect.map", encoding='UTF-8')
    word_dic_file = open("./Data/"+config.dataset_name+"/word.senti.map", encoding='UTF-8')
    # -----Global Input data u:user/a:aspect/i:item/w:word-----

    aspect_dic = {}
    word_senti_dic = {}
    word_dic = {}
    # user to construct R
    ui_rating_dic = {}
    ui_rating_dic_test = {}
    # user to construct S
    uia_senti_dic_train = {}
    uia_senti_dic_test = {}
    # # user to construct O
    iaw_frequency_dic = {}
    # user to calculate BPR
    iaw_frequency_test_dic = {}
    uw_frequency_mat = np.zeros((U_num, W_num))

    ia_list = list(range(I_num))  # 列表数组
    uiaw_list = []
    if_pair_list = []  # 一对一对

    # -----Global Generated data-----
    # create aspects and words mapping data
    for line in aspect_dic_file.readlines():
        eachline = line.strip().split('=')
        aspect_dic[int(eachline[0])] = eachline[1]
    for line in word_dic_file.readlines():
        eachline = line.strip().split('=')
        word_dic[eachline[0]] = eachline[1]
        word_senti_dic[int(eachline[0])] = int(eachline[2])

    cnt_train = 0
    cnt_test = 0

    # read user/item-feture-word entries（2）构建UFO和UFW稀疏张量
    for line in uiaw_train_file.readlines():
        cnt_train += 1
        line = line.replace("\n", "")
        eachline = line.strip().split("\t")  # yelp是空格
        u_idx = int(eachline[0])
        i_idx = int(eachline[1])
        # rating = ui_rating_dic[str([u_idx, i_idx])]
        a_idx = int(eachline[2])
        w_idx = int(eachline[3])
        rating = float(eachline[4])
        w_senti = word_senti_dic[w_idx]
        uiaw_list.append(str([u_idx, i_idx, a_idx, w_idx]))

        ui_rating_dic[str([u_idx, i_idx])] = rating
        uw_frequency_mat[u_idx][w_idx] += 1
        if str([i_idx, a_idx, w_idx]) not in iaw_frequency_dic:
            iaw_frequency_dic[str([i_idx, a_idx, w_idx])] = 0
        iaw_frequency_dic[str([i_idx, a_idx, w_idx])] += 1
        if str([u_idx, i_idx, a_idx]) not in uia_senti_dic_train:
            uia_senti_dic_train[str([u_idx, i_idx, a_idx])] = 0
        uia_senti_dic_train[str([u_idx, i_idx, a_idx])] += w_senti

    for line in uiaw_test_file.readlines():
        cnt_test += 1
        line = line.replace("\n", "")
        eachline = line.strip().split("\t")
        u_idx = int(eachline[0])
        i_idx = int(eachline[1])
        a_idx = int(eachline[2])
        w_idx = int(eachline[3])
        rating = float(eachline[4])
        w_senti = word_senti_dic[w_idx]
        ui_rating_dic_test[str([u_idx, i_idx])] = rating
        if str([i_idx, a_idx, w_idx]) not in iaw_frequency_test_dic:
            iaw_frequency_test_dic[str([i_idx, a_idx, w_idx])] = 0
        iaw_frequency_test_dic[str([i_idx, a_idx, w_idx])] += 1
        if str([u_idx, i_idx, a_idx]) not in uia_senti_dic_test:
            uia_senti_dic_test[str([u_idx, i_idx, a_idx])] = 0
        uia_senti_dic_test[str([u_idx, i_idx, a_idx])] += w_senti
    # sigmoid处理
    for key in uia_senti_dic_train.keys():
        uia_senti_dic_train[key] = 1 + 4 / (1 + np.exp(0 - uia_senti_dic_train[key]))
    for key in uia_senti_dic_test.keys():
        uia_senti_dic_test[key] = 1 + 4 / (1 + np.exp(0 - uia_senti_dic_test[key]))
    for key in iaw_frequency_dic.keys():
        index = get_index(key)
        x = iaw_frequency_dic[key] / 20
        iaw_frequency_dic[key] = 5 * (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        if word_senti_dic[index[2]] < 0:
            iaw_frequency_dic[key] = 0  # 极性反转或消除负面词
        if iaw_frequency_dic[key] < 0.5:
            iaw_frequency_dic[key] = 0  # 处理低频词

    print("train size:" + str(cnt_train))
    print("test size:" + str(cnt_test) + '\n')

    uiaw_train_file.close()
    uiaw_test_file.close()
    aspect_dic_file.close()
    word_dic_file.close()
    return uiaw_list, uw_frequency_mat, ui_rating_dic, uia_senti_dic_train, iaw_frequency_dic, ui_rating_dic_test, word_dic, \
           aspect_dic, iaw_frequency_test_dic,uia_senti_dic_test
