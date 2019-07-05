with open("./uiawr_id.entry","w")as f2:
    with open("./uiawr.entry","r")as f:
        for line in f.readlines():
            line = line.split("\t")
            user = line[0]
            item = line[1]
            aspect = line[2]
            word = line[3]
            rating = float(line[4])+1
            f2.write("{}\t{}\t{}\t{}\t{}".format(user, item, aspect, word, rating))
            f2.write("\n")