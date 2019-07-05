global dataset_name
global num_iter
global print_every_times


class config:
    dataset_name = "dianping"
    num_iter = 8000
    print_every_times = 5000  # every200
    U_dim = 32
    I_dim = 16
    F_dim = 16
    W_dim = 32
    minibatch = 400
    lmd_reg = 8
    lmd_r = 1
    lmd_s = 0.3  # 0.35
    lmd_o = 0.8  # 0.87
    lr = 0.1
