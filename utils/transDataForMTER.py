import numpy as np

U_num = 10719
I_num = 10410
F_num = 104
# -----Input file-----r:rating/s:sentiment/ u:user/a:aspect/i:item/w:word
uiaw_train_file = open('../Data/dianping/uiawr_id.train', encoding='UTF-8')
uiaw_test_file = open('../Data/dianping/uiawr_id.test', encoding='UTF-8')
word_senti_file = open('../Data/dianping/word.senti.map', encoding='UTF-8')
# -----Output file-----
of_uiaw_train_entry = open("../TransformedData/for_MTER/dianping_uiaw_train.entry", "a", encoding='UTF-8')
of_uiaw_test_entry = open("../TransformedData/for_MTER/dianping_uiaw_test.entry", "a", encoding='UTF-8')

train_entry = {}
test_entry = {}

word_dic ={}
word_senti_dic = {}
for line in word_senti_file.readlines():
    eachline = line.strip().split('=')
    word_dic[eachline[0]] = eachline[1]
    word_senti_dic[int(eachline[0])] = int(eachline[2])

for line in uiaw_train_file.readlines():
    line = line.replace("\n", "")
    eachline = line.strip().split("\t")
    u_idx = int(eachline[0])
    i_idx = int(eachline[1])
    a_idx = int(eachline[2])
    w_idx = int(eachline[3])
    word_senti = word_senti_dic[w_idx]
    rating = int(float(eachline[4]))
    of_uiaw_train_entry.write(str(u_idx) + "," + str(i_idx) + "," + str(a_idx) + "," + str(w_idx))
    of_uiaw_train_entry.write("\n")

    if str(u_idx) + "," + str(i_idx) + "," + str(rating) not in train_entry:
        train_entry[str(u_idx) + "," + str(i_idx) + "," + str(rating)] = []
    train_entry[str(u_idx) + "," + str(i_idx) + "," + str(rating)].append("{}:{}".format(a_idx, word_senti))
for line in uiaw_test_file.readlines():
    line = line.replace("\n", "")
    eachline = line.strip().split("\t")
    u_idx = int(eachline[0])
    i_idx = int(eachline[1])
    a_idx = int(eachline[2])
    w_idx = int(eachline[3])
    rating = int(float(eachline[4]))
    of_uiaw_test_entry.write(str(u_idx) + "," + str(i_idx) + "," + str(a_idx) + "," + str(w_idx))
    of_uiaw_test_entry.write("\n")

    if str(u_idx) + "," + str(i_idx) + "," + str(rating) not in test_entry:
        test_entry[str(u_idx) + "," + str(i_idx) + "," + str(rating)] = []
    test_entry[str(u_idx) + "," + str(i_idx) + "," + str(rating)].append("{}:{}".format(a_idx, word_senti))

uiaw_train_file.close()
uiaw_test_file.close()
of_uiaw_train_entry.close()
of_uiaw_test_entry .close()

# write file:
of_train_entry = open("../TransformedData/for_MTER/dianping_train.entry", "w")
of_test_entry = open("../TransformedData/for_MTER/dianping_test.entry", "w")
for key in train_entry:
    of_train_entry.write(key)
    of_train_entry.write(",")
    for item in train_entry[key]:
        of_train_entry.write(item+" ")
    of_train_entry.write("\n")
for key in test_entry:
    of_test_entry.write(key)
    of_test_entry.write(",")
    for item in test_entry[key]:
        of_test_entry.write(item+" ")
    of_test_entry.write("\n")
