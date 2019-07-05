ui_set = set()
with open("./uiawr_id.entry","r")as f:
    for line in f.readlines():
        line = line.replace("\n","")
        line = line.split("\t")
        user_id = line[0]
        item_id = line[1]
        ui_set.add((user_id,item_id))

print(ui_set.__len__())