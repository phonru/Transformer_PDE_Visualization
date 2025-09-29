import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Subset

def h5_data_read(filename):
    file = h5py.File(filename, 'r', )
    u = file['u'][:,].astype('float32')
    f = file['f'][:,].astype('float32')
    re = file['re'][:,].astype('float32')
    c = file['c'][:,].astype('float32')
    return u, f, re, c

class InitDataset_sample(Dataset):
    def __init__(self, filename):#, device):
        self.file = h5py.File(filename, 'r', )
        # self.len = self.file['re'][:].shape[0]
        self.len = self.file['re'].shape[0]
        # u, f, c, re = h5_data_read(filename)
        # self.u = u#[:, :, :, :, :, :]
        # self.f = f#[:, :, :, :, :, :]
        # self.c = c#[:, :, :, :, :, :]
        # self.re = re
        # self.u = torch.Tensor(u)[:, :, :, :, :, :]
        # self.f = torch.Tensor(f)[:, :, :, :, :, :]
        # self.c = torch.Tensor(c)[:, :, :, :, :, :]
        # self.re = torch.Tensor(re)
        # self.device = device
        # umax = torch.max(self.u)
        # umin = torch.min(self.u)
        # fmax = torch.max(self.f)
        # fmin = torch.min(self.f)
        # print('a')
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        u = self.file['u'][idx,].astype('float32')
        f = self.file['f'][idx,].astype('float32')
        c = self.file['c'][idx,].astype('float32')
        re = self.file['re'][idx,].astype('float32')
        
        # u = self.u[idx, :, :, :, :, :]#.to(self.device)
        # f = self.f[idx, :, :, :, :, :]#.to(self.device)
        # c = self.c[idx, :, :, :, :, :]#.to(self.device)
        # gamma = self.re[idx, ]#.to(self.device)
        return u, f, c, re
    
class InitDatasetWithLatent(InitDataset_sample):
    """读取 (u, f, c, re, z)"""
    def __init__(self, filename):#, device):
        self.file = h5py.File(filename, 'r', )
        # self.len = self.file['re'][:].shape[0]
        self.len = self.file['re'].shape[0]
    def __getitem__(self, idx):
        u, f, c, re = super().__getitem__(idx)
        z = self.file['z'][idx].astype('float32')
        return u, f, c, re, z     # 直接把 z 当 label
    
    
    
    
class InitDataset(Dataset):
    def __init__(self, filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.files = [h5py.File(fn, 'r') for fn in filenames]
        self.lens = [f['c'].shape[0] for f in self.files]
        self.cumsum = np.cumsum([0] + self.lens)
        self.total_len = sum(self.lens)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 找到属于哪个文件
        file_idx = np.searchsorted(self.cumsum, idx, side='right') - 1
        local_idx = idx - self.cumsum[file_idx]
        f = self.files[file_idx]
        u = f['u'][local_idx].astype('float32')
        f_ = f['f'][local_idx].astype('float32')
        c = f['c'][local_idx].astype('float32')
        re = f['re'][local_idx].astype('float32')
        return u, f_, c, re




# device = torch.device('cuda:4')
# a = InitDataset('/home/gqc/dataset/mu3_dataset_32.h5', device)

# class FastH5Dataset(Dataset):
#     def __init__(self, filename):
#         self.filename = filename
#         self.file = None
#         self.len = None

#     def _init_file_if_needed(self):
#         if self.file is None:
#             # 每个 worker 进程都会单独打开一份文件句柄
#             self.file = h5py.File(self.filename, 'r')
#             self.len = self.file['re'].shape[0]

#     def __len__(self):
#         if self.len is None:
#             # 在主进程 __len__ 调用时，只打开一次元信息
#             with h5py.File(self.filename, 'r') as f:
#                 self.len = f['re'].shape[0]
#         return self.len

#     def __getitem__(self, idx):
#         # 这里 _getitem_ 只把 idx 返回，让 collate_fn 去实际读取
#         return idx

# def fast_h5_collate(ids,dataset):
#     """
#     ids: 当前 batch 的索引列表，例如 [3, 17, 25, 42]
#     通过合并连续区段，减少随机 I/O 调用次数
#     """
#     # 支持 Subset
#     if isinstance(dataset, Subset):
#         base_dataset = dataset.dataset
#         # ids 已经是原始数据集的索引
#         mapped_ids = ids
#     else:
#         base_dataset = dataset
#         mapped_ids = ids
    
#     # 全局 dataset 用于访问文件（闭包捕获全局变量）
#     base_dataset._init_file_if_needed()
#     file = base_dataset.file

#     sorted_ids = sorted(ids)
#     segments = []
#     start = sorted_ids[0]
#     prev = start
#     for i in sorted_ids[1:]:
#         if i == prev + 1:
#             prev = i
#         else:
#             segments.append((start, prev))
#             start = i
#             prev = i
#     segments.append((start, prev))

#     # 暂存每个 idx 对应的数组
#     batch_u, batch_f, batch_c, batch_re = {}, {}, {}, {}
#     for (s0, s1) in segments:
#         # 读取 [s0 : s1+1]
#         u_data = file['u'][s0:s1+1]
#         f_data = file['f'][s0:s1+1]
#         c_data = file['c'][s0:s1+1]
#         re_data = file['re'][s0:s1+1]
#         # 注意 data 已经是 float32，无需再 astype
#         length = s1 - s0 + 1
#         for offset in range(length):
#             idx = s0 + offset
#             batch_u[idx]  = u_data[offset]
#             batch_f[idx]  = f_data[offset]
#             batch_c[idx]  = c_data[offset]
#             batch_re[idx] = re_data[offset]

#     # # 按照原来 ids 顺序组织成 list
#     # ordered_u = [batch_u[i] for i in ids]
#     # ordered_f = [batch_f[i] for i in ids]
#     # ordered_c = [batch_c[i] for i in ids]
#     # ordered_re = [batch_re[i] for i in ids]
    
#     # 按照原 ids 顺序组织（注意是 mapped_ids 的顺序）
#     ordered_u = [batch_u[i] for i in mapped_ids]
#     ordered_f = [batch_f[i] for i in mapped_ids]
#     ordered_c = [batch_c[i] for i in mapped_ids]
#     ordered_re = [batch_re[i] for i in mapped_ids]

#     # 最后用 default_collate 转成 Tensor，或根据需求自己拼接
#     return default_collate(ordered_u), default_collate(ordered_f), \
#            default_collate(ordered_c), default_collate(ordered_re)

