infile = "../Data/dianping/uiawr_id.entry"


f = open("../TransformedData/for_otherMF/dianping_ratings.txt", "w")
ui_alreadys = set()
with open(infile, "r") as infile_:
    for line in infile_.readlines():
        line = line.replace("\n", "")
        eachline = line.strip().split("\t")  # yelp 是空格
        u_idx = eachline[0]
        i_idx = eachline[1]
        if (u_idx, i_idx) in ui_alreadys:
            continue
        ui_alreadys.add((u_idx, i_idx))
        rating = float(eachline[4])
        f.write(u_idx + " " + i_idx + " " + str(rating))
        f.write("\n")
f.close()
