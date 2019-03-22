import autograd.numpy as np
from autograd import multigrad
import time
from Core.Loss import minibath_sparse_cost


def construct_IF_W_vec(I, F, W, if_pair):
    A = np.hstack((I[if_pair[0]], F[if_pair[1]]))
    # 某个产品的某个特征的词向量
    if_W = np.einsum("a,ma->m", A, W)
    return if_W


def construct_IF_U_row(U, I, F, u_id, i_id, item_f_pair_list2):
    # 某用户对某产品各个特征F的评分
    U_if_all = {}
    for a in range(len(item_f_pair_list2)):
        A = np.hstack((I[i_id], F[item_f_pair_list2[a]]))
        U_if_all[item_f_pair_list2[a]] = np.einsum("a,a->", A, U[u_id])
    return U_if_all


def minibatch_adagradSGD_train(uiaw_list, uw_frequency_mat,
                               ui_rating_dic, uia_senti_dic, iaw_frequency_dic,
                               U_dim, I_dim, F_dim, W_dim, U_num, I_num, F_num_1more, W_num, num_iter,
                               lmd_reg, lmd_r,lmd_s, lmd_o, neg_sample_rate, lmd_bpr, minibatch,
                               lr,random_seed=0, eps=1e-8, ):
    np.random.seed(random_seed)
    cost = minibath_sparse_cost

    U_dim_initial = (U_num, U_dim)
    I_dim_initial = (I_num, I_dim)
    F_dim_initial = (F_num_1more, F_dim)
    W_dim_initial = (W_num, W_dim)

    U = np.random.rand(*U_dim_initial)
    I = np.random.rand(*I_dim_initial)
    F = np.random.rand(*F_dim_initial)
    W = np.random.rand(*W_dim_initial)

    sum_square_gradients_U = np.zeros_like(U)
    sum_square_gradients_I = np.zeros_like(I)
    sum_square_gradients_F = np.zeros_like(F)
    sum_square_gradients_W = np.zeros_like(W)

    mg = multigrad(cost, argnums=[0, 1, 2, 3])

    # SGD procedure
    for i in range(num_iter):
        starttime = time.time()
        print(i + 1)
        del_u, del_i, del_f, del_w = mg(U, I, F, W, uiaw_list, uw_frequency_mat,
                                        ui_rating_dic, uia_senti_dic, iaw_frequency_dic,
                                        lmd_reg, lmd_r,lmd_s, lmd_o, neg_sample_rate, lmd_bpr, minibatch)
        # eps+del_g**2
        sum_square_gradients_U += eps + np.square(del_u)
        sum_square_gradients_I += eps + np.square(del_i)
        sum_square_gradients_F += eps + np.square(del_f)
        sum_square_gradients_W += eps + np.square(del_w)

        # np.divide()对位除法只保留整数部分，np.sqrt()各元素平方根 lr=0.1，# 0.1/((eps+del_g**2)**1/2)
        lr_u = np.divide(lr, np.sqrt(sum_square_gradients_U))
        lr_i = np.divide(lr, np.sqrt(sum_square_gradients_I))
        lr_f = np.divide(lr, np.sqrt(sum_square_gradients_F))
        lr_w = np.divide(lr, np.sqrt(sum_square_gradients_W))

        # 自适应梯度下降 G1=G1 - 0.1/(adagrad**1/2) * del_g
        U -= lr_u * del_u
        I -= lr_i * del_i
        F -= lr_f * del_f
        W -= lr_w * del_w

        # Projection to non-negative space
        U[U < 0] = 0
        I[I < 0] = 0
        F[F < 0] = 0
        W[W < 0] = 0

        nowtime = time.time()
        timeleft = (nowtime - starttime) * (num_iter - i - 1)

        if timeleft / 60 > 60:
            print('time left: ' + str(int(timeleft / 3600)) + ' hr ' + str(int(timeleft / 60 % 60)) + ' min ' + str(
                int(timeleft % 60)) + ' s')
        else:
            print("time left: " + str(int(timeleft / 60)) + ' min ' + str(int(timeleft % 60)) + ' s')

    return U, I, F, W
