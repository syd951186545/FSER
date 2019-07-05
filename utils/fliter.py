user_hasreviews = {}
item_hasusers = {}
aspect_freq = {}
word_freq = {}
with open("../Data/dianping/uiawr.all.entry", "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        line = line.split("\t")
        u_ = line[0]
        i_ = line[1]
        a_ = line[2]
        w_ = line[3]
        rating = line[4]
        user_hasreviews.setdefault(u_, [])
        user_hasreviews[u_].append(i_)

        item_hasusers.setdefault(i_, [])
        item_hasusers[i_].append(i_)
        if a_ not in aspect_freq:
            aspect_freq[a_] = 0
        aspect_freq[a_] += 1
        if w_ not in word_freq:
            word_freq[w_] = 0
        word_freq[w_] += 1
user_few_hasreviews = []
item_few_hasusers = []
aspect_many_freq = []
word_many_freq = []

for aspect in aspect_freq.keys():
    if aspect_freq[aspect] > 3000:
        aspect_many_freq.append(aspect)
for word in word_freq.keys():
    if word_freq[word] > 50:
        word_many_freq.append(word)


# 删除低频产品
for item in item_hasusers.keys():
    if len(item_hasusers[item]) < 35:
        item_few_hasusers.append(item)
for item in item_few_hasusers:
    del item_hasusers[item]
# for item in item_hasusers.values():
#     for user in user_few_hasreviews:
#         if user in item:
#             item.remove(user)

# 删除低频用户
for user in user_hasreviews.values():
    for item in item_few_hasusers:
        if item in user:
            user.remove(item)
# for user in user_few_hasreviews:
#     del user_hasreviews[user]

# for user in user_hasreviews.keys():
#     if len(user_hasreviews[user]) < 0:
#         user_few_hasreviews.append(user)
#         unstop = True
#     else:
#         unstop = False
num_reviews1 = 0
for item_users in item_hasusers.values():
    num_reviews1 += len(item_users)
num_reviews2 = 0
for users_item in user_hasreviews.values():
    num_reviews2 += len(users_item)


rf = open("../Data/dianping/uiawr.all.entry", "r", encoding="utf-8")
with open("../Data/dianping/uiawr.wt_order.entry", "w", encoding="utf-8") as wf:
    for line in rf.readlines():
        s = line
        line = line.replace("\n","")
        line = line.split("\t")
        u_ = line[0]
        i_ = line[1]
        a_ = line[2]
        w_ = line[3]
        rating = line[4]
        if u_ in user_hasreviews and i_ in item_hasusers and a_ in aspect_many_freq and w_ in word_many_freq:
            wf.write(s)



