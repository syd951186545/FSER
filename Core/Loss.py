import autograd.numpy as np
import numpy
import random as rdm


def cos_sim(A_, B_):
    # num = A_ * B_.transpose()
    num = np.einsum("a,a->", A_, B_)
    denom = np.linalg.norm(A_) * np.linalg.norm(B_)
    if denom == 0:
        raise Exception("Denominator !be 0")
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


def minibath_sparse_cost(U, I, F, W, uiaw_list, uw_frequency_mat,
                         ui_rating_dic, uia_senti_dic, iaw_frequency_dic,
                         lmd_reg, lmd_r, lmd_s, lmd_o, neg_sample_rate, lmd_bpr, minibatch):
    loss_R = 0
    loss_S = 0
    loss_O = 0
    loss_bpr = 0
    lmd_reg = lmd_reg
    lmd_r = lmd_r
    lmd_s = lmd_s
    lmd_o = lmd_o
    neg_sample_rate = neg_sample_rate
    lmd_bpr = lmd_bpr
    minibatch = minibatch

    uiaw_sample_list = rdm.sample(uiaw_list, minibatch)
    # uiaw_sample_list = ["[0,0,0,0]", ]
    for uiaw in uiaw_sample_list:
        uiaw = uiaw[1:-1].split(',')
        u_id = int(uiaw[0])
        i_id = int(uiaw[1])
        a_id = int(uiaw[2])
        w_id = int(uiaw[3])
        real_rating = ui_rating_dic[str([u_id, i_id])]
        real_uia_senti = uia_senti_dic[str([u_id, i_id, a_id])]
        real_iaw_frequency = iaw_frequency_dic[str([i_id, a_id, w_id])]
        _A = np.hstack((I[i_id], F[-1]))
        loss_R += (real_rating - np.einsum("a,a->", U[u_id], _A)) ** 2
        A_ = np.hstack((I[i_id], F[a_id]))
        loss_S += (real_uia_senti - np.einsum("a,a->", U[u_id], A_)) ** 2
        if real_iaw_frequency > 0:
            loss_O += (real_iaw_frequency - np.einsum("a,a->", W[w_id], A_)) ** 2

        value_i = cos_sim(W[w_id], U[u_id])
        # neg_sample
        if numpy.random.random() > neg_sample_rate:
            w_id2 = numpy.random.choice(numpy.nonzero(uw_frequency_mat[u_id])[0])
            j = 0
            while uw_frequency_mat[u_id][w_id2] == uw_frequency_mat[u_id][w_id] and j < 100:
                w_id2 = numpy.random.choice(numpy.nonzero(uw_frequency_mat[u_id])[0])
                j += 1
            if uw_frequency_mat[u_id][w_id2] == uw_frequency_mat[u_id][w_id]:
                continue
        else:
            w_id2 = numpy.random.choice(numpy.where(uw_frequency_mat[u_id] == 0)[0])
        value_j = cos_sim(W[w_id2], U[u_id])
        try:
            sign = (uw_frequency_mat[u_id][w_id] - uw_frequency_mat[u_id][w_id2]) / np.abs(
                uw_frequency_mat[u_id][w_id] - uw_frequency_mat[u_id][w_id2])
            loss_bpr += -sign * (0 - np.log(1 / (1 + np.exp(-sign * (value_i - value_j)))))
        except ArithmeticError:
            print("Denominator !be 0")

    # loss_R /= minibatch
    # loss_S /= minibatch
    # loss_O /= minibatch
    # loss_bpr /= minibatch

    loss_Regularization = 0
    error = U.flatten()
    loss_Regularization += np.sqrt((error ** 2).mean())
    error = I.flatten()
    loss_Regularization += np.sqrt((error ** 2).mean())
    error = F.flatten()
    loss_Regularization += np.sqrt((error ** 2).mean())
    error = W.flatten()
    loss_Regularization += np.sqrt((error ** 2).mean())

    print('loss_R:')
    print(loss_R)
    print('loss_S:')
    print(loss_S)
    print('loss_O:')
    print(loss_O)
    print('loss_bpr:')
    print(loss_bpr)
    print("Total lost:")
    print(lmd_r * loss_R + lmd_s * loss_S + lmd_o * loss_O + lmd_bpr * loss_bpr + lmd_reg * loss_Regularization)

    return lmd_r * loss_R + lmd_s * loss_S + lmd_o * loss_O + lmd_bpr * loss_bpr + lmd_reg * loss_Regularization


if __name__ == "__main__":
    A = np.array([0, 1])
    B = np.array([0, -1])
    a = cos_sim(A, B)
    print(a)
