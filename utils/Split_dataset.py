import random

ui_set = set()
u_set = set()
with open("../Data/dianping/uiawr_id.entry", "r", encoding="UTF-8")as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        line = line.split("\t")
        u_id = line[0]
        i_id = line[1]
        ui_set.add((u_id, i_id))
        u_set.add(u_id)
ui_set_train = set()
ui_set_test = set()
for ui in ui_set:
    # if ui[0] not in u_set:
    #     ui_set_train.add(ui)
    if random.random() < 0.8:
        ui_set_train.add(ui)
    else:
        ui_set_test.add(ui)


train_file = open("../Data/dianping/uiawr_id.train", "w", encoding="UTF-8")
test_file = open("../Data/dianping/uiawr_id.test", "w", encoding="UTF-8")
with open("../Data/dianping/uiawr_id.entry", "r", encoding="UTF-8")as file:
    for line in file.readlines():
        s = line
        line = line.replace("\n", "")
        line = line.split("\t")
        u_id = line[0]
        i_id = line[1]
        if (u_id, i_id) in ui_set_train:
            train_file.write(s)
        else:
            test_file.write(s)



train_file.close()
test_file.close()