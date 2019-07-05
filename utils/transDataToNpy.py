import numpy as np
# yelp
# U_num = 10719
# I_num = 10410
# F_num = 104
# dianping
U_num = 11753
I_num = 17241
F_num = 135
# -----Input file-----r:rating/s:sentiment/ u:user/a:aspect/i:item/w:word
uiaw_train_file = open('../Data/dianping/uiawr_id.train', encoding='UTF-8')
uiaw_test_file = open('../Data/dianping/uiawr_id.test', encoding='UTF-8')

rating_train = np.zeros((U_num, I_num))
rating_test = np.zeros((U_num, I_num))
item_feature_mentioned = np.zeros((I_num, F_num))
item_feature_mentioned_test = np.zeros((I_num, F_num))
user_feature_mentioned = np.zeros((U_num, F_num))
user_feature_mentioned_test = np.zeros((U_num, F_num))

for line in uiaw_train_file.readlines():
    line = line.replace("\n", "")
    eachline = line.strip().split("\t")
    u_idx = int(eachline[0])
    i_idx = int(eachline[1])
    a_idx = int(eachline[2])
    rating = float(eachline[4])
    item_feature_mentioned[i_idx][a_idx] += 1
    user_feature_mentioned[u_idx][a_idx] += 1
    if rating_train[u_idx][i_idx] ==0:
        rating_train[u_idx][i_idx] = rating
for line in uiaw_test_file.readlines():
    line = line.replace("\n", "")
    eachline = line.strip().split("\t")
    u_idx = int(eachline[0])
    i_idx = int(eachline[1])
    a_idx = int(eachline[2])
    rating = float(eachline[4])
    item_feature_mentioned_test[i_idx][a_idx] += 1
    user_feature_mentioned_test[u_idx][a_idx] += 1
    if rating_test[u_idx][i_idx] == 0:
        rating_test[u_idx][i_idx] = rating


indexs = item_feature_mentioned.nonzero()
indexs_2 = item_feature_mentioned_test.nonzero()
index_u1 = user_feature_mentioned.nonzero()
index_u2 = user_feature_mentioned_test.nonzero()
# for index in item_feature_mentioned.nonzero():
item_feature_mentioned[indexs] = 1 + 4 / (1 + np.exp(-item_feature_mentioned[indexs]))
item_feature_mentioned_test[indexs_2] = 1 + 4 / (1 + np.exp(-item_feature_mentioned_test[indexs_2]))
user_feature_mentioned[index_u1] = 1 + 4*(2/(1 + np.exp(-user_feature_mentioned[index_u1]))-1)
user_feature_mentioned_test[index_u2] = 1 + 4*(2/(1 + np.exp(-user_feature_mentioned_test[index_u2]))-1)
np.save("../TransformedData/for_EFM/dianping_ia_mentioned_train", item_feature_mentioned)
np.save("../TransformedData/for_EFM/dianping_ia_mentioned_test", item_feature_mentioned_test)
np.save("../TransformedData/for_EFM/dianping_ua_mentioned_train", user_feature_mentioned)
np.save("../TransformedData/for_EFM/dianping_ua_mentioned_test", user_feature_mentioned_test)
np.save("../TransformedData/for_EFM/dianping_rating_train", rating_train)
np.save("../TransformedData/for_EFM/dianping_rating_test", rating_test)
