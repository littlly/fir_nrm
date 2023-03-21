class settings(object):
    def __init__(self, debug = False):
        self.day = 2
        self.xcentral = 181
        self.ycentral = 141
        self.radius = 10 * self.day

        self.step = 5 #every step number to select a sample
        self.input_horizon = 100
        self.output_horizon = 24 * self.day
        self.lr = 1e-6

        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.activation = 'relu'

        self.bitch_size = 256
        self.num_epoch = 3
        self.use_gpu = True
        self.debug = False

        self.data_file='/media/cweg/c555f7e8-d283-4f4c-8f94-5997ca3e98a9/rmf/adaptor.mars.internal-1669533684.269949-25455-10-a450af95-380d-4cd3-a782-b24202b940dc.nc'
        self.log_path = '/media/cweg/c555f7e8-d283-4f4c-8f94-5997ca3e98a9/rmf/model_norm_new'

        self.seed = 2020012595