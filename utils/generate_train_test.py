import shutil

f01 = open("uiawr_id.01", "r")
f02 = open("uiawr_id.02", "r")
f03 = open("uiawr_id.03", "r")
f04 = open("uiawr_id.04", "r")
f05 = open("uiawr_id.05", "r")
test = "f05"
ftrain = open("uiawr_id.train", "a")
ftest = "uiawr_id.test"

f = ["f01", "f02", "f03", "f04", "f05"]
for file in f:
    if file == test:
        shutil.copyfile("uiawr_id."+file[1:], ftest)
    else:
        for line in eval(file).readlines():
            line = line.replace("\n", "")
            eachline = line.strip().split(" ")
            u_idx = eachline[0]
            i_idx = eachline[1]
            a_idx = eachline[2]
            w_idx = eachline[3]
            rating = float(eachline[4])
            ftrain.write(u_idx + " " + i_idx + " " + a_idx + " " + w_idx + " " + str(rating) + "\n")
