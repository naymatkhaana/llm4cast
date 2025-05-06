import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from data_provider.datam import get_series_and_dates
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        self.data_x = []
        self.data_y = []
        self.data_stamp = []

        #print("########################################### self.args.foundation_setting",self.args.foundation_setting)
        if self.args.foundation_setting == 1 and self.set_type == 0:
            self.data_paths = [ "national_illness_age0.csv", "national_illness_age5.csv", "national_illness_nop.csv", "national_illness_ot.csv", "national_illness_uw.csv", "national_illness_w.csv", "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]
            #self.data_paths = [ "covid_Alabama.csv", "covid_Alaska.csv",  "covid_Arizona.csv", "covid_California.csv", "covid_Colorado.csv", "covid_Connecticut.csv", "covid_Delaware.csv",  "covid_District of Columbia.csv",   "covid_Guam.csv", "covid_Hawaii.csv", "covid_Idaho.csv", "covid_Illinois.csv", "covid_Indiana.csv", "covid_Iowa.csv", "covid_Kansas.csv", "covid_Louisiana.csv", "covid_Maine.csv",  "covid_Massachusetts.csv", "covid_Michigan.csv", "covid_Minnesota.csv", "covid_Mississippi.csv", "covid_Missouri.csv", "covid_Montana.csv", "covid_Nebraska.csv", "covid_Nevada.csv", "covid_New Hampshire.csv", "covid_New Jersey.csv", "covid_New Mexico.csv", "covid_New York.csv", "covid_North Carolina.csv", "covid_Northern Mariana Islands.csv", "covid_Ohio.csv",  "covid_Oregon.csv", "covid_Pennsylvania.csv", "covid_Puerto Rico.csv", "covid_Rhode Island.csv",  "covid_South Dakota.csv", "covid_Tennessee.csv",  "covid_Virgin Islands.csv", "covid_Virginia.csv", "covid_Washington.csv", "covid_West Virginia.csv", "covid_Wisconsin.csv", "covid_Wyoming.csv" ]
        else:
            self.data_paths = [self.data_path]

        for file_path in self.data_paths:

            df_raw = pd.read_csv(os.path.join(self.root_path, file_path))
            if "ILI" in file_path:
                df_raw = df_raw.iloc[:471]

            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]


            num_train = int(len(df_raw) * (0.8))
            num_test = int(len(df_raw) * (1-(0.8)))
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x.append( data[border1:border2] )
            self.data_y.append( data[border1:border2] )

            #if self.set_type == 0 and self.args.augmentation_ratio > 0:
            #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

            self.data_stamp.append( data_stamp  )

    def __getitem__(self, index):

        if self.args.is_data_covid == 1 or self.set_type != 0 or self.args.foundation_setting == 0:
            ind = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)      

            s_begin = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[ind][s_begin:s_end]
            seq_y = self.data_y[ind][r_begin:r_end]
            seq_x_mark = self.data_stamp[ind][s_begin:s_end]
            seq_y_mark = self.data_stamp[ind][r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

        else:
            ind = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)      
            if ind < 6:
                s_begin = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = self.data_x[ind][s_begin:s_end]
                seq_y = self.data_y[ind][r_begin:r_end]
                seq_x_mark = self.data_stamp[ind][s_begin:s_end]
                seq_y_mark = self.data_stamp[ind][r_begin:r_end]

                return seq_x, seq_y, seq_x_mark, seq_y_mark

            else:
                #print("######################################################## index",index)

                #print("########################################################  (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)", (len(self.data_x[0]) - self.seq_len - self.pred_len + 1))

                index = index - ((len(self.data_x[0]) - self.seq_len - self.pred_len + 1)*6)
                ind =6 + ( index // (len(self.data_x[7]) - self.seq_len - self.pred_len + 1)  )
                s_begin = index % (len(self.data_x[7]) - self.seq_len - self.pred_len + 1)
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                #print("######################################################## ind",ind)
                #print("######################################################## index",index)

                #print("######################################################## len(self.data_x)",len(self.data_x))

                #print("######################################################## len(self.data_x[0])",len(self.data_x[0]))

                #print("######################################################## len(self.data_x[6])",len(self.data_x[6]))

                seq_x = self.data_x[ind][s_begin:s_end]
                seq_y = self.data_y[ind][r_begin:r_end]
                seq_x_mark = self.data_stamp[ind][s_begin:s_end]
                seq_y_mark = self.data_stamp[ind][r_begin:r_end]

                return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        if self.args.is_data_covid == 1 or self.set_type != 0 or self.args.foundation_setting == 0:
            return  (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * (len(self.data_paths))

        return  (len(self.data_x[7]) - self.seq_len - self.pred_len + 1) * (len(self.data_paths)-6) + (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * (6)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class Dataset_CustomOrig(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.2)
        num_train = 1025; num_test = 118

        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        #self.scale = False



        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            #train_data = df_data[border1s[0]:border2s[0]]
            #self.scaler.fit(train_data.values)
            data = df_data.values
        if self.args.with_pre_norm == 0:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_CustomOrig_fewshot(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        df_raw = df_raw.iloc[457:,:]
        num_train = int(len(df_raw) * 0.4)
        #print("###############################################",num_train)
        num_test = int(len(df_raw) * 0.6)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #print("################### ", len(self.data_x),self.pred_len, self.seq_len, len(self.data_x) - self.seq_len - self.pred_len + 1)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)






class Dataset_CustomOrigInf(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_x = []; self.data_y = []; self.data_stamp = []
        print("#######################", self.set_type)

        if self.set_type == 0:
            #self.data_paths = [ "national_illness_age0.csv", "national_illness_age5.csv", "national_illness_nop.csv", "national_illness_ot.csv", "national_illness_uw.csv", "national_illness_w.csv", "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]
            self.data_paths = [ "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]

        else:
            self.data_paths = [self.data_path]

        print(self.data_paths)
        for file_path in self.data_paths:
			
            df_raw = pd.read_csv(os.path.join(self.root_path, file_path))
            print(os.path.join(self.root_path, file_path))
            
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            num_train = int(len(df_raw) * 0.457)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.features == 'M' or self.features == 'MS':
            	cols_data = df_raw.columns[1:]
            	df_data = df_raw[cols_data]
            elif self.features == 'S':
            	df_data = df_raw[[self.target]]

            #self.scale = False
            if self.scale:
            	train_data = df_data[border1s[0]:border2s[0]]
            	self.scaler.fit(train_data.values)
            	data = self.scaler.transform(df_data.values)
            else:
                #train_data = df_data[border1s[0]:border2s[0]]
                #self.scaler.fit(train_data.values)
                data = df_data.values
            
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
            	df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            	df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            	df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            	df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            	data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
            	data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            	data_stamp = data_stamp.transpose(1, 0)
            
            self.data_x.append(data[border1:border2])
            self.data_y.append(data[border1:border2])
            
            #if self.set_type == 0 and self.args.augmentation_ratio > 0:
            #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
            self.data_stamp.append(data_stamp)
		    
    def __getitem__(self, index):

        ind = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        s_begin = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        #s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[ind][s_begin:s_end]
        seq_y = self.data_y[ind][r_begin:r_end]
        seq_x_mark = self.data_stamp[ind][s_begin:s_end]
        seq_y_mark = self.data_stamp[ind][r_begin:r_end]

        #rint("############# seq_y.shape",seq_y.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * len(self.data_paths)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_CustomOrigInf_fewshot(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_x = []; self.data_y = []; self.data_stamp = []
        print("#######################", self.set_type)

        if self.set_type == 0:
            #self.data_paths = [ "national_illness_age0.csv", "national_illness_age5.csv", "national_illness_nop.csv", "national_illness_ot.csv", "national_illness_uw.csv", "national_illness_w.csv", "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]
            self.data_paths = [ "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]

        else:
            self.data_paths = [self.data_path]

        print(self.data_paths)
        for file_path in self.data_paths:
			
            df_raw = pd.read_csv(os.path.join(self.root_path, file_path))
            print(os.path.join(self.root_path, file_path))
            
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            num_train = int(len(df_raw) * 0.8)
            if self.set_type == 0: 
                num_train = int(len(df_raw) * 0.1465)

            num_test = int(len(df_raw) * 0.2)

            if self.set_type == 2: 
                num_train = int(len(df_raw) * 0.2)
                num_test = int(len(df_raw) * 0.8)

            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.features == 'M' or self.features == 'MS':
            	cols_data = df_raw.columns[1:]
            	df_data = df_raw[cols_data]
            elif self.features == 'S':
            	df_data = df_raw[[self.target]]
            
            if self.scale:
            	train_data = df_data[border1s[0]:border2s[0]]
            	self.scaler.fit(train_data.values)
            	data = self.scaler.transform(df_data.values)
            else:
            	data = df_data.values
            
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
            	df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            	df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            	df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            	df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            	data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
            	data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            	data_stamp = data_stamp.transpose(1, 0)
            
            self.data_x.append(data[border1:border2])
            self.data_y.append(data[border1:border2])
            
            #if self.set_type == 0 and self.args.augmentation_ratio > 0:
            #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
            self.data_stamp.append(data_stamp)
		    
    def __getitem__(self, index):

        ind = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        s_begin = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        #s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[ind][s_begin:s_end]
        seq_y = self.data_y[ind][r_begin:r_end]
        seq_x_mark = self.data_stamp[ind][s_begin:s_end]
        seq_y_mark = self.data_stamp[ind][r_begin:r_end]

        #rint("############# seq_y.shape",seq_y.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * len(self.data_paths)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_datam(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_files = [  'electricity_weekly_dataset',  'fred_md_dataset',  'hospital_dataset',  'kdd_cup_2018_dataset_without_missing_values',    'm1_monthly_dataset',  'm1_quarterly_dataset',  'm1_yearly_dataset',  'm3_monthly_dataset', 'm3_quarterly_dataset',  'm3_yearly_dataset',  'm4_daily_dataset',  'm4_hourly_dataset',  'm4_monthly_dataset',  'm4_quarterly_dataset',  'm4_weekly_dataset',   'pedestrian_counts_dataset',  'rideshare_dataset_without_missing_values',  'saugeenday_dataset',  'solar_10_minutes_dataset',  'solar_4_seconds_dataset',  'solar_weekly_dataset',  'sunspot_dataset_without_missing_values',  'temperature_rain_dataset_without_missing_values',  'tourism_monthly_dataset',  'tourism_quarterly_dataset',  'tourism_yearly_dataset',  'traffic_hourly_dataset',  'traffic_weekly_dataset',  'us_births_dataset',  'vehicle_trips_dataset_without_missing_values', 'wind_4_seconds_dataset',  'wind_farms_minutely_dataset_without_missing_values']   
        #
        #'car_parts_dataset_without_missing_values',  
        #, 'kaggle_web_traffic_dataset_without_missing_values'
        #'london_smart_meters_dataset_without_missing_values',
        # '  'm3_other_dataset', '
        #'m4_yearly_dataset', 
        #' 'weather_dataset','

        self.__read_data__()


    def __read_data__(self):
        self.data_x = []; self.data_y = []; self.data_stamp = []; self.data_lengths = []
        
        if self.set_type == 2 or self.set_type == 1:
            df_read = pd.read_csv(os.path.join(self.root_path,self.data_path))
            self.__add_data__(df_read) 
            self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
            self.cumsum = np.cumsum(self.dlengths)
            return

        for dfile in self.data_files:
            print("************* dfile: ", dfile)
            diter = 0
            for dseries, dates in get_series_and_dates("dataset/datam/"+dfile+"/"+dfile+".tsf"):
                print(diter); diter+=1;
                df_read = pd.DataFrame()
                df_read['OT'] = dseries
                df_read['date'] = dates
            
                self.__add_data__(df_read)  
            print("############### len(self.data_lengths): ",len(self.data_lengths))    
            print("############### np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
            print("############### np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      

        print("############### Total len(self.data_lengths): ",len(self.data_lengths))    
        print("############### Total np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
        print("############### Total np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      
        self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
        self.cumsum = np.cumsum(self.dlengths)


    def __add_data__(self,df_raw):
        self.scaler = StandardScaler()
        #df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * (1 if self.set_type == 0 else 0.7))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x.append(data[border1:border2])
        self.data_y.append(data[border1:border2])
        self.data_lengths.append(len(data[border1:border2]))

        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp.append(data_stamp)

    def __getitem__(self, index):

        

        #it = 0
        #ind = index
        #cumsum = dlengths[it]
        
        #while cumsum <= ind:
        #    it+=1
        #    cumsum += dlengths[it]
            
        # example dlengths: [6, 4, 6, 3, 3, 5]    , ind: 10
        
        # Calculate the cumulative sum of lengths
    
        it = np.searchsorted(self.cumsum, index, side='right')
    
        s_begin = self.dlengths[it] - (self.cumsum[it] - index)

        #s_begin =  self.dlengths[it] - (self.cumsum[it]-index) #index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[it][s_begin:s_end]
        seq_y = self.data_y[it][r_begin:r_end]
        seq_x_mark = self.data_stamp[it][s_begin:s_end]
        seq_y_mark = self.data_stamp[it][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)










class Dataset_datam_short(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_files = [  'electricity_weekly_dataset',  'fred_md_dataset',  'hospital_dataset',  'kdd_cup_2018_dataset_without_missing_values',    'm1_monthly_dataset',  'm1_quarterly_dataset',  'm1_yearly_dataset',  'm3_monthly_dataset', 'm3_quarterly_dataset',  'm3_yearly_dataset',  'm4_daily_dataset',  'm4_hourly_dataset',  'm4_monthly_dataset',  'm4_quarterly_dataset',  'm4_weekly_dataset',   'pedestrian_counts_dataset',  'rideshare_dataset_without_missing_values',  'saugeenday_dataset',  'solar_10_minutes_dataset',  'solar_4_seconds_dataset',  'solar_weekly_dataset',  'sunspot_dataset_without_missing_values',  'temperature_rain_dataset_without_missing_values',  'tourism_monthly_dataset',  'tourism_quarterly_dataset',  'tourism_yearly_dataset',  'traffic_hourly_dataset',  'traffic_weekly_dataset',  'us_births_dataset',  'vehicle_trips_dataset_without_missing_values', 'wind_4_seconds_dataset',  'wind_farms_minutely_dataset_without_missing_values']   
        #
        #'car_parts_dataset_without_missing_values',  
        #, 'kaggle_web_traffic_dataset_without_missing_values'
        #'london_smart_meters_dataset_without_missing_values',
        # '  'm3_other_dataset', '
        #'m4_yearly_dataset', 
        #' 'weather_dataset','

        self.__read_data__()


    def __read_data__(self):
        self.data_x = []; self.data_y = []; self.data_stamp = []; self.data_lengths = []
        
        if self.set_type == 2 or self.set_type == 1:
            df_read = pd.read_csv(os.path.join(self.root_path,self.data_path))
            self.__add_data__(df_read) 
            self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
            self.cumsum = np.cumsum(self.dlengths)
            return

        for dfile in self.data_files:
            print("************* dfile: ", dfile)
            diter = 0
            for dseries, dates in get_series_and_dates("dataset/datam/"+dfile+"/"+dfile+".tsf"):
                print(diter); diter+=1;
                df_read = pd.DataFrame()
                df_read['OT'] = dseries
                df_read['date'] = dates
            
                self.__add_data__(df_read)  
            print("############### len(self.data_lengths): ",len(self.data_lengths))    
            print("############### np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
            print("############### np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      

        print("############### Total len(self.data_lengths): ",len(self.data_lengths))    
        print("############### Total np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
        print("############### Total np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      
        self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
        self.cumsum = np.cumsum(self.dlengths)
        self.total_len = np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1)
        self.sample_arr = np.random.randint(self.total_len, size=(int(self.total_len*0.01),))



    def __add_data__(self,df_raw):
        self.scaler = StandardScaler()
        #df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * (1 if self.set_type == 0 else 0.7))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x.append(data[border1:border2])
        self.data_y.append(data[border1:border2])
        self.data_lengths.append(len(data[border1:border2]))

        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp.append(data_stamp)

    def __getitem__(self, index):

        

        #it = 0
        #ind = index
        #cumsum = dlengths[it]
        
        #while cumsum <= ind:
        #    it+=1
        #    cumsum += dlengths[it]
            
        # example dlengths: [6, 4, 6, 3, 3, 5]    , ind: 10
        
        # Calculate the cumulative sum of lengths

        if self.set_type == 0: 
            index = self.sample_arr[index] #np.random.randint(self.total_len, size=(1,))[0]
    
        it = np.searchsorted(self.cumsum, index, side='right')
    
        s_begin = self.dlengths[it] - (self.cumsum[it] - index)

        #s_begin =  self.dlengths[it] - (self.cumsum[it]-index) #index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[it][s_begin:s_end]
        seq_y = self.data_y[it][r_begin:r_end]
        seq_x_mark = self.data_stamp[it][s_begin:s_end]
        seq_y_mark = self.data_stamp[it][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):

        if self.set_type == 0: 
            return int(self.total_len*0.01)

        else:
            return np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_CustomOrigInfv(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.cum_sum = np.cumsum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1 )

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_x = []; self.data_y = []; self.data_stamp = []; self.data_lengths = []
        print("#######################", self.set_type)

        if self.set_type == 0:
            ###self.data_paths = [ "national_illness_age0.csv", "national_illness_age5.csv", "national_illness_nop.csv", "national_illness_ot.csv", "national_illness_uw.csv", "national_illness_w.csv", "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]
            self.data_paths = [ "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]

        else:
            self.data_paths = [self.data_path]

        print(self.data_paths)
        for file_path in self.data_paths:
			
            df_raw = pd.read_csv(os.path.join(self.root_path, file_path))
            print(os.path.join(self.root_path, file_path))
            
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            num_train = int(len(df_raw) * 0.8)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.features == 'M' or self.features == 'MS':
            	cols_data = df_raw.columns[1:]
            	df_data = df_raw[cols_data]
            elif self.features == 'S':
            	df_data = df_raw[[self.target]]

            #self.scale = False
            if self.scale:
            	train_data = df_data[border1s[0]:border2s[0]]
            	self.scaler.fit(train_data.values)
            	data = self.scaler.transform(df_data.values)
            else:
                #train_data = df_data[border1s[0]:border2s[0]]
                #self.scaler.fit(train_data.values)
                data = df_data.values
            
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
            	df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            	df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            	df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            	df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            	data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
            	data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            	data_stamp = data_stamp.transpose(1, 0)
            
            self.data_x.append(data[border1:border2])
            self.data_y.append(data[border1:border2])
            self.data_lengths.append(border2-border1)
            
            #if self.set_type == 0 and self.args.augmentation_ratio > 0:
            #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
            self.data_stamp.append(data_stamp)
		    
    def __getitem__(self, index):

        ##ind = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        ##s_begin = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        #s_begin = index
  
        it = np.searchsorted(self.cum_sum, index, side='right')
    
        s_begin = (self.data_lengths[it] - self.seq_len - self.pred_len + 1) - (self.cum_sum[it] - index)

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[it][s_begin:s_end]
        seq_y = self.data_y[it][r_begin:r_end]
        seq_x_mark = self.data_stamp[it][s_begin:s_end]
        seq_y_mark = self.data_stamp[it][r_begin:r_end]

        #print("############# seq_x.shape",seq_x.shape, "############# seq_y.shape",seq_y.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #return (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * len(self.data_paths)
        return np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1 ) 
        #(len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * len(self.data_paths)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_CustomOrigInfv2(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.cum_sum = np.cumsum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1 )

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_x = []; self.data_y = []; self.data_stamp = []; self.data_lengths = []
        print("#######################", self.set_type)

        if self.set_type == 0: #"national_illness_24.csv",
            ###self.data_paths = [ "national_illness_age0.csv", "national_illness_age5.csv", "national_illness_nop.csv", "national_illness_ot.csv", "national_illness_uw.csv", "national_illness_w.csv", "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]
            #self.data_paths = [ "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"]
            self.data_paths = [ "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv","covid_national_weekly.csv", "covid_Alabama.csv", "covid_Alaska.csv",  "covid_Arizona.csv", "covid_California.csv", "covid_Colorado.csv", "covid_Connecticut.csv", "covid_Delaware.csv",  "covid_District of Columbia.csv",   "covid_Guam.csv", "covid_Hawaii.csv", "covid_Idaho.csv", "covid_Illinois.csv", "covid_Indiana.csv", "covid_Iowa.csv", "covid_Kansas.csv", "covid_Louisiana.csv", "covid_Maine.csv",  "covid_Massachusetts.csv", "covid_Michigan.csv", "covid_Minnesota.csv", "covid_Mississippi.csv", "covid_Missouri.csv", "covid_Montana.csv", "covid_Nebraska.csv", "covid_Nevada.csv", "covid_New Hampshire.csv", "covid_New Jersey.csv", "covid_New Mexico.csv", "covid_New York.csv", "covid_North Carolina.csv", "covid_Northern Mariana Islands.csv", "covid_Ohio.csv",  "covid_Oregon.csv", "covid_Pennsylvania.csv", "covid_Puerto Rico.csv", "covid_Rhode Island.csv",  "covid_South Dakota.csv", "covid_Tennessee.csv",  "covid_Virgin Islands.csv", "covid_Virginia.csv", "covid_Washington.csv", "covid_West Virginia.csv", "covid_Wisconsin.csv", "covid_Wyoming.csv"]
            #self.data_paths = [ "national_illness_24.csv", "ILI_tn.csv",  "ILI_ar.csv" , "ILI_id.csv",  "ILI_la.csv",  "ILI_mn.csv",  "ILI_ne.csv",  "ILI_mi.csv",  "ILI_wv.csv",  "ILI_nh.csv" , "ILI_nj.csv" ,  "ILI_nd.csv"  ,"ILI_sd.csv" ,  "ILI_de.csv" ,  "ILI_ks.csv",   "ILI_ky.csv",   "ILI_nv.csv" , "ILI_ut.csv"  , "ILI_va.csv"   ,"ILI_vt.csv" , "ILI_ri.csv", "ILI_tx.csv"] #,"covid_national_weekly.csv", "covid_Alabama.csv", "covid_Alaska.csv",  "covid_Arizona.csv", "covid_California.csv", "covid_Colorado.csv", "covid_Connecticut.csv", "covid_Delaware.csv",  "covid_District of Columbia.csv",   "covid_Guam.csv", "covid_Hawaii.csv", "covid_Idaho.csv", "covid_Illinois.csv", "covid_Indiana.csv", "covid_Iowa.csv", "covid_Kansas.csv", "covid_Louisiana.csv", "covid_Maine.csv",  "covid_Massachusetts.csv", "covid_Michigan.csv", "covid_Minnesota.csv", "covid_Mississippi.csv", "covid_Missouri.csv", "covid_Montana.csv", "covid_Nebraska.csv", "covid_Nevada.csv", "covid_New Hampshire.csv", "covid_New Jersey.csv", "covid_New Mexico.csv", "covid_New York.csv", "covid_North Carolina.csv", "covid_Northern Mariana Islands.csv", "covid_Ohio.csv",  "covid_Oregon.csv", "covid_Pennsylvania.csv", "covid_Puerto Rico.csv", "covid_Rhode Island.csv",  "covid_South Dakota.csv", "covid_Tennessee.csv",  "covid_Virgin Islands.csv", "covid_Virginia.csv", "covid_Washington.csv", "covid_West Virginia.csv", "covid_Wisconsin.csv", "covid_Wyoming.csv"]

        else:
            self.data_paths = [self.data_path]

        print(self.data_paths)
        for file_path in self.data_paths:
			
            df_raw = pd.read_csv(os.path.join(self.root_path, file_path))
            print(os.path.join(self.root_path, file_path))
            
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            
            cols = list(df_raw.columns)
            #if self.target in cols:
            #    cols.remove(self.target)
            cols.remove('date')
            for col in cols:

                df_raw = df_raw[['date'] + cols] #+ [self.target]]
                #num_train = int(len(df_raw) * 0.9)
                #num_test = int(len(df_raw) * 0.1)
                if "covid" in file_path:
                    num_train = 115; num_test = 42
                if "national_illness_24" in file_path:
                    num_train = 1025; num_test = 118 #num_train = 1025-171; num_test = 171+118 #num_train = 1025; num_test = 118
                if "ILI" in file_path:
                    num_train = 606; num_test = 82 #num_train = 606-171; num_test = 171+82 #num_train = 606; num_test = 82

                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
            
                #if self.features == 'M' or self.features == 'MS':
               	#    cols_data = df_raw.columns[1:]
            	#    df_data = df_raw[cols_data]
                #elif self.features == 'S':
            	#    df_data = df_raw[[self.target]]

                df_data = df_raw[[col]]
                print("#################, df_data.shape", df_data.shape, "col",col)

                if self.scale:
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    #train_data = df_data[border1s[0]:border2s[0]]
                    #self.scaler.fit(train_data.values)
                    data = df_data.values
                if self.args.with_pre_norm == 0:
                    data = df_data.values
            
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                if self.timeenc == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], 1).values
                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)
                print("#################, data[border1:border2].shape", data[border1:border2].shape)
                self.data_x.append(data[border1:border2])
                self.data_y.append(data[border1:border2])
                self.data_lengths.append(border2-border1)
            
                #if self.set_type == 0 and self.args.augmentation_ratio > 0:
                #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
                self.data_stamp.append(data_stamp)
		    
    def __getitem__(self, index):

        ##ind = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        ##s_begin = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        #s_begin = index
  
        it = np.searchsorted(self.cum_sum, index, side='right')
    
        s_begin = (self.data_lengths[it] - self.seq_len - self.pred_len + 1) - (self.cum_sum[it] - index)

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[it][s_begin:s_end]
        seq_y = self.data_y[it][r_begin:r_end]
        seq_x_mark = self.data_stamp[it][s_begin:s_end]
        seq_y_mark = self.data_stamp[it][r_begin:r_end]

        #print("############# seq_x.shape",seq_x.shape, "############# seq_y.shape",seq_y.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #return (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * len(self.data_paths)
        return np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1 ) 
        #(len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * len(self.data_paths)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



