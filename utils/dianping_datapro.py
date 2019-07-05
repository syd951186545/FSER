def pro_raw_data():
    dianping_data = "../Data/dianping/dianping.txt"

    user_num_dic = {}
    itemm_num_dic = {}
    word_senti_map = {}
    uiawr = []
    with open(dianping_data, "r", encoding="utf-8")as df:
        i = 0
        for line in df.readlines():

            line = line.replace("\n", "")
            if len(line) == 0:
                continue
            if line[0] == '<':
                continue

            s = line.split("\t")
            if s[0].isdigit():

                user_id = s[0]
                item_id = s[1]
                rating = float(int(s[2])+int(s[3])+int(s[4]))/3
                rating = round(rating, 2)
                user_num_dic.setdefault(user_id, 0)
                itemm_num_dic.setdefault(item_id, 0)

                user_num_dic[user_id] += 1
                itemm_num_dic[item_id] += 1

            if line[0] == "[" and (line[-2:] == "N]" or line[-2:] == "Y]"):
                line = line.replace("\n", "")
                line = line.replace("\n", "")
                line = line.split("\t")
                for aws in line:
                    aws = aws.replace("\'", "")
                    aws = aws.replace("[", "")
                    aws = aws.replace("]", "")
                    aws = aws.split(",")
                    aws = list(aws)
                    a_ = aws[0].replace(" ", "")
                    w_ = aws[1].replace(" ", "")
                    s_ = aws[2]
                    uiawr.append([user_id, item_id, a_, w_, str(rating)])
                    word_senti_map[w_] = int(s_)
            i += 1
    with open("../Data/dianping/uiawr.all.entry", "w", encoding="utf-8") as f:
        for uiawr1 in uiawr:
            f.write("{}\t{}\t{}\t{}\t{}".format(uiawr1[0], uiawr1[1], uiawr1[2], uiawr1[3], uiawr1[4]))
            f.write("\n")
        # with open("../Data/dianping/word.senti.map", "w", encoding="utf-8") as f2:
        #     for i, word in zip(range(len(word_senti_map)), word_senti_map.keys()):
        #         f2.write("{}={}={}".format(i, word, word_senti_map[word]))
        #         f2.write("\n")


def set_map():
    user_set = set()
    item_set = set()
    aspect_set = set()
    word_set = set()

    user_dic = {}
    item_dic = {}
    aspect_dic = {}
    word_dic = {}
    with open("../Data/dianping/uiawr.wt_order.entry", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            line = line.split("\t")
            user_set.add(line[0])
            item_set.add(line[1])
            aspect_set.add(line[2])
            word_set.add(line[3])
        with open("../Data/dianping/user.map", "w", encoding="utf-8") as f2:
            for i, user in zip(range(len(user_set)), [user for user in user_set]):
                user_dic[user] = i
                f2.write("{}={}".format(i, user))
                f2.write("\n")
        with open("../Data/dianping/item.map", "w", encoding="utf-8") as f3:
            for i, item in zip(range(len(item_set)), [item for item in item_set]):
                item_dic[item] = i
                f3.write("{}={}".format(i, item))
                f3.write("\n")
        with open("../Data/dianping/aspect.map", "w", encoding="utf-8") as f4:
            for i, aspect in zip(range(len(aspect_set)), [aspect for aspect in aspect_set]):
                aspect_dic[aspect] = i
                f4.write("{}={}".format(i, aspect))
                f4.write("\n")
        for i, word in zip(range(len(word_set)), [word for word in word_set]):
            word_dic[word] = i

    with open("../Data/dianping/uiawr.wt_order.entry", "r", encoding="utf-8") as rrf:
        with open("../Data/dianping/uiawr.entry", "w", encoding="utf-8") as wwf:
            for line in rrf.readlines():
                line = line.replace("\n", "")
                line = line.split("\t")
                user = line[0]
                item = line[1]
                aspect = line[2]
                word = line[3]
                rating = line[4]

                user__ = user_dic[user]
                item__ = item_dic[item]
                aspect__ = aspect_dic[aspect]
                word__ = word_dic[word]
                wwf.write("{}\t{}\t{}\t{}\t{}".format(user__, item__, aspect__, word__, rating))
                wwf.write("\n")
    word_senti_dic = {}
    with open("../Data/dianping/word.senti_all.map", "r", encoding="utf-8")as wordf:
        for line in wordf.readlines():
            line = line.replace("\n", "")
            line = line.split("=")
            word = line[1]
            senti = line[2]
            word_senti_dic[word] = senti
        with open("../Data/dianping/word.senti.map", "w", encoding="utf-8")as wordnf:
            for word in word_dic:
                wordnf.write("{}={}={}".format(word_dic[word], word, word_senti_dic[word]))
                wordnf.write("\n")


if __name__ == '__main__':
    # pro_raw_data()
    set_map()
