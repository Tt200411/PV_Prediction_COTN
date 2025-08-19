import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_PV(Dataset):
    """
    专门为光伏数据设计的数据加载器
    修复了train/vali/test分割逻辑问题
    """
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='Site_1_50MW.csv', 
                 target='Power', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 96      # 24小时历史
            self.label_len = 48    # 12小时起始
            self.pred_len = 100    # 预测100个时间步（25小时）
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # 验证flag - 添加pred支持
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train':0, 'val':1, 'test':2, 'pred':2}  # pred使用test的分割
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 检查数据格式
        print(f"加载数据: {self.data_path}")
        print(f"数据形状: {df_raw.shape}")
        print(f"列名: {df_raw.columns.tolist()}")
        
        # 修复光伏数据分割逻辑 - 使用80%/10%/10%分割
        data_len = len(df_raw)
        train_end = int(data_len * 0.8)
        val_end = int(data_len * 0.9)
        
        # 确保有足够的数据进行序列预测
        min_required = self.seq_len + self.pred_len
        
        # 重新计算边界，确保每个集合都有足够的数据
        border1s = [0, 
                   max(0, train_end - self.seq_len), 
                   max(0, val_end - self.seq_len)]
        border2s = [train_end, 
                   val_end, 
                   data_len]
        
        # 验证数据分割合理性
        for i, (b1, b2) in enumerate(zip(border1s, border2s)):
            available_samples = max(0, b2 - b1 - self.seq_len - self.pred_len + 1)
            set_name = ['train', 'val', 'test'][i]
            print(f"{set_name}: {b1} -> {b2} ({b2-b1} 点, {available_samples} 样本)")
            
            if available_samples <= 0:
                print(f"警告: {set_name}集合样本数不足，可能影响训练")
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 特征选择
        if self.features=='M' or self.features=='MS':
            # 多变量：排除日期列
            cols_data = [col for col in df_raw.columns if col != 'date']
            df_data = df_raw[cols_data]
        elif self.features=='S':
            # 单变量：只使用目标列
            df_data = df_raw[[self.target]]
        
        # 标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # 时间特征
        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        else:
            # 如果没有日期列，创建虚拟时间戳
            data_stamp = np.zeros((border2-border1, 1))

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], 
                                  self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# 保持原有ETT数据加载器以便兼容
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init - 添加pred支持
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train':0, 'val':1, 'test':2, 'pred':2}  # pred使用test的分割
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], 
                                  self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_provider(args, flag):
    """
    数据提供器 - 自动选择合适的数据加载器
    """
    Data = Dataset_PV if 'Site_' in args.data else Dataset_ETT_hour
    timeenc = 0 if args.embed!='timeF' else 1
    
    if flag == 'test':
        shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
    elif flag=='pred':
        shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        Data = Dataset_PV if 'Site_' in args.data else Dataset_ETT_hour
    else:
        shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader