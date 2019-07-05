import random
from operator import methodcaller

f1 = "E:\PYworkspace\FSER\Data\\uiawr_id.train"
f2 = "E:\PYworkspace\FSER\Data\\uiawr_id.test"
f01 = open("uiawr_id.01", "w")
f02 = open("uiawr_id.02", "w")
f03 = open("uiawr_id.03", "w")
f04 = open("uiawr_id.04", "w")
f05 = open("uiawr_id.05", "w")

with open(f1, "r") as f1_, open(f2, "r") as f2_:
    for line in f1_.readlines():
        line = line.replace("\n", "")
        eachline = line.strip().split(" ")
        u_idx = eachline[0]
        i_idx = eachline[1]
        a_idx = eachline[2]
        w_idx = eachline[3]
        rating = float(eachline[4])
        s = random.choice(["f01", "f02", "f03", "f04", "f05"])
        eval(s).write(u_idx + " " + i_idx + " " + a_idx + " " + w_idx + " " + str(rating) + "\n")
    for line in f2_.readlines():
        line = line.replace("\n", "")
        eachline = line.strip().split(" ")
        u_idx = eachline[0]
        i_idx = eachline[1]
        a_idx = eachline[2]
        w_idx = eachline[3]
        rating = float(eachline[4])
        s = random.choice(["f01", "f02", "f03", "f04", "f05"])
        eval(s).write(u_idx + " " + i_idx + " " + a_idx + " " + w_idx + " " + str(rating) + "\n")

f01.close()
f02.close()
f03.close()
f04.close()
f05.close()
