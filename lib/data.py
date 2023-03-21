import numpy as np
import xarray as xr

from seetings import settings

class Data(object):

    def __init__(self, debug = False):
        """
        Inputs
        ======
        :param data_file: input csv filepath.
        :param input_horizon:
        :param n_stations:
        :param train_ratio:
        :param debug:"""

        #super(Data, self).__init__(debug= False)

        self.s = settings()

        self.trainDataRate = 0.6 if not debug else 0.1 # percentage of data used for training (saving time for debuging)

        self.u10, self.v10, self.t2m, self.msl = Data.read_data(self)
        self.u10, self.u10means_stds = Data.normalize_data(self.u10)
        self.v10, self.v10means_stds = Data.normalize_data(self.v10)
        self.t2m, self.t2mmeans_stds = Data.normalize_data(self.t2m)
        self.msl, self.mslmeans_stds = Data.normalize_data(self.t2m)


    @staticmethod
    def read_data(self):
        with xr.open_dataset(self.s.data_file) as f:

            u10 = f['u10'].data[: , (self.s.xcentral - self.s.radius-1):(self.s.xcentral + self.s.radius),
            (self.s.ycentral - self.s.radius-1):(self.s.ycentral + self.s.radius)].astype(np.float32)
            v10 = f['v10'].data[: , (self.s.xcentral - self.s.radius-1):(self.s.xcentral + self.s.radius),
            (self.s.ycentral - self.s.radius-1):(self.s.ycentral + self.s.radius)].astype(np.float32)
            t2m = f['t2m'].data[: , (self.s.xcentral - self.s.radius-1):(self.s.xcentral + self.s.radius),
            (self.s.ycentral - self.s.radius-1):(self.s.ycentral + self.s.radius)].astype(np.float16)
            msl = f['msl'].data[: , (self.s.xcentral - self.s.radius-1):(self.s.xcentral + self.s.radius),
            (self.s.ycentral - self.s.radius-1):(self.s.ycentral + self.s.radius)].astype(np.float32)

        return u10, v10, t2m, msl

    @staticmethod
    def normalize_data(winds):
        wind_min = winds.min()
        wind_max = winds.max() - wind_min

        normal_winds = (winds - wind_min) / wind_max
        mins_maxs = [wind_min, wind_max]

        return normal_winds, mins_maxs

    def denormalize_data(self, u10vec):
        uwind_min, uwind_max = self.u10means_stds#the number have the min and max from original files
        ures = u10vec * uwind_max + uwind_min
        #vwind_min, vwind_max = self.v10means_stds#the number have the min and max from original files
        #vres = v10vec * vwind_max + vwind_min
        #the above two functions are used to one-dimension all datas
        #vec is the predict proportion
        return ures

    def load_data_cnn(self): # For CNN

        samples = []
        ys = []
        num_ele = np.ceil((self.u10.shape[0] - self.s.output_horizon - self.s.input_horizon) / self.s.step)
        for index in range(int(num_ele)):

            samples.append(np.stack((self.u10[index * self.s.step : index * self.s.step + self.s.input_horizon +1,:,:],
            self.v10[index * self.s.step : index * self.s.step + self.s.input_horizon +1,:,:],
            self.t2m[index * self.s.step : index * self.s.step + self.s.input_horizon +1,:,:],
            self.msl[index * self.s.step : index * self.s.step + self.s.input_horizon +1,:,:])))

            ys.append(np.stack((self.u10[index * self.s.step + self.s.input_horizon + 1:index * self.s.step + self.s.input_horizon +self.s.output_horizon +1,self.s.radius+1,self.s.radius+1],
            self.v10[index * self.s.step + self.s.input_horizon + 1:index * self.s.step + self.s.input_horizon +self.s.output_horizon + 1,self.s.radius+1,self.s.radius+1])))
                
        #in sample 0 dimension = the numbers of examples,numbers; 1 dimension =4 channels; 2 is time; 3 is xarray; 4 is yarray

        samples = np.array(samples)
        ys = np.array(ys)

        n_train_samples = int(np.ceil(num_ele * self.s.train_ratio))
        n_val_samples = int(np.ceil(num_ele * (self.s.val_ratio + self.s.train_ratio)))

        X_train = samples[: n_train_samples]
        y_train = ys[: n_train_samples]#[examples,uv,output_horizon]

        X_test = samples[n_val_samples : int(num_ele)]
        y_test = ys[n_val_samples : int(num_ele)]

        X_val = samples[n_train_samples : n_val_samples]
        y_val = ys[n_train_samples : n_val_samples]

        return [X_train, y_train], [X_val, y_val], [X_test, y_test]
        #X_train[0] is a array, but X_train is a list

#    def load_data(self, pre_x_train_val, pre_y_train_val, model, rnn_model_num): # pre_x_train_val and pre_y_train_val from load_data_lstm_1

        X_train_val, y_train_val = np.ones_like(pre_x_train_val), np.zeros_like(pre_y_train_val)

        for ind in range(len(pre_x_train_val) - 1):
            tempInput = pre_x_train_val[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1, temp_shape[0], temp_shape[1]))

            output = model.predict(rnn_model_num, tempInput)

            tInput = np.reshape(tempInput, temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)

            X_train_val[ind] = tempInput
            y_train_val[ind] = pre_y_train_val[ind + 1]

        X_train_val = X_train_val[:-1]
        y_train_val = y_train_val[:-1]

        return [X_train_val, y_train_val]
