import numpy as np


# ---------Output or Not------------
# ---------Reverse mapping----------
def Output(nowTime, U, I, F, W, U_num, iaw_frequency_dic, word_dic, aspect_dic, rec_item, ui_rating_dic_test,iaw_frequency_test_dic):
    ia_exist_dic = {}
    iaw_frequency_dic.update(iaw_frequency_test_dic)
    for iaw_s in iaw_frequency_dic.keys():
        iaw_s = iaw_s[1:-1].split(",")
        i_id = iaw_s[0]
        a_id = iaw_s[1]
        if i_id in ia_exist_dic.keys():
            ia_exist_dic[i_id].append(a_id)
        else:
            ia_exist_dic[i_id] = [a_id, ]
        ia_exist_dic[i_id] = list(set(ia_exist_dic[i_id]))
    # word2id_dic = {}
    # aspect2id_dic = {}
    # for item in word_dic.items():
    #     word2id_dic[item[1]] = item[0]
    # for item in aspect_dic.items():
    #     aspect2id_dic[item[1]] = item[0]
    # -----Output file------------------
    print('Output to file...')
    item_recrank_file = open("./Result/" + str(nowTime) + ".reclist", "w", encoding="UTF-8")
    item_recexplian_file = open("./Result/" + str(nowTime) + ".explanation", "w", encoding="UTF-8")
    rank_length = 100
    top_word_num = 6
    top_aspect_num = 7
    for user_id in range(U_num):
        user_item_rec_rank = []
        item_rec_list = np.copy(rec_item[user_id])
        for item_length in range(rank_length):
            # top @rank_length pre_rating and their item_id
            item_id = np.argmax(item_rec_list)
            key = str([user_id, item_id])
            user_item_rec_rank.append(key)
            item_rec_list[item_id] = -1
        if not set(user_item_rec_rank).isdisjoint(set(list(ui_rating_dic_test.keys()))):
            # recitem is in test file
            # interset = set(user_item_rec_rank).intersection(set(list(ui_rating_dic_test.keys())))
            item_recrank_file.write("@USER:" + str(user_id) + "\n")
            item_recexplian_file.write("@USER:" + str(user_id) + " ")
            for ui in user_item_rec_rank:
                iid = int(ui[1:-1].split(",")[1])
                pre_rating = rec_item[user_id][iid]
                item_recrank_file.write(str(iid) + " ")
                if str([user_id, iid]) in ui_rating_dic_test.keys():
                    realRating = ui_rating_dic_test[str([user_id, iid])]
                    item_recrank_file.write(str(pre_rating) + " " + "realRating:" + str(realRating) + "\n")
                    # 对于产品iid，他的全部aspect，计算sentiment和word_frequency
                else:
                    item_recrank_file.write(str(pre_rating) + "\n")

                item_recexplian_file.write("@ITEM:" + str(iid) + "\n")
                dic = {}
                ia_exist_ = []
                if str(iid) not in ia_exist_dic:
                    continue
                for aspect2 in ia_exist_dic[str(iid)]:
                    A_ = np.hstack((I[iid], F[int(aspect2)]))
                    pre_aspect_senti2 = np.einsum("a,a->", U[user_id], A_)
                    dic[pre_aspect_senti2] = aspect2
                pre_senti_aspect = list(dic.keys())
                pre_senti_aspect.sort(reverse=True)
                for senti in pre_senti_aspect[:top_aspect_num]:
                    ia_exist_.append(dic[senti])
                for aspect in ia_exist_:
                    item_recexplian_file.write(aspect_dic[int(aspect)])
                    item_recexplian_file.write(":")
                    A_ = np.hstack((I[iid], F[int(aspect)]))
                    pre_aspect_senti = np.einsum("a,a->", U[user_id], A_)
                    pre_aspect_senti = round(float(pre_aspect_senti), 4)
                    item_recexplian_file.write(str(pre_aspect_senti))
                    pre_word = np.einsum("a,ma->m", A_, W)
                    # pre_word = np.einsum("a,ma->m", U[user_id], W)
                    wn = 0
                    while wn < top_word_num:
                        w_id = pre_word.argmax()
                        if pre_word[w_id] == -100:
                            break
                        elif str([iid, int(aspect), w_id]) in iaw_frequency_dic.keys():
                            word = word_dic[str(w_id)]
                            word_freq = pre_word[w_id]
                            word_freq = round(word_freq, 4)
                            item_recexplian_file.write(" " + word + " " + str(word_freq))
                            wn += 1
                        pre_word[w_id] = -100
                    item_recexplian_file.write("\n")


    item_recrank_file.close()
    item_recexplian_file.close()
    print("Output is just OK!")

