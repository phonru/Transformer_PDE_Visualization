
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Subset
from itertools import zip_longest

from tqdm import tqdm


def h5_data_read(filename):
    file = h5py.File(filename, 'r', )
    u = file['u'][:,].astype('float32')
    f = file['f'][:,].astype('float32')
    re = file['re'][:,].astype('float32')
    c = file['c'][:,].astype('float32')
    return u, f, re, c

class InitDataset_sample(Dataset):
    def __init__(self, filename, size, ):
        self.file = h5py.File(filename, 'r', )
        self.len = self.file['re'].shape[0]
 
        if size == self.len:
            pass
        else:
            raise ValueError(f"Provided size {size} does not match dataset length {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        u = self.file['u_in'][idx,].astype('float32')   # (B=1,C=2,T=32,32,32)
        f = self.file['f'][idx,].astype('float32')
        c = self.file['c'][idx,].astype('float32')
        re = self.file['re'][idx,].astype('float32')

        u_gt = self.file['u_out'][idx,].astype('float32') 

        return u, f, c, re, u_gt
    
  
# def setup_multi_dataloader(filename,sizes,batch_sizes):
    
#     datasets = [InitDataset_sample(filename[idx], sz) for idx, sz in enumerate(sizes)]
    
#     loaders = [
#         DataLoader(
#             dataset=ds,
#             batch_size=bs,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True
#         )
#         for ds, bs in zip(datasets, batch_sizes)
#     ]
#     # print(f"Batch sizes for each dataset: {batch_sizes}")
#     return loaders
      
    
if __name__ == "__main__":
    # —— 1. 定义子数据集 —— #
    # —— 2. 构造 2个子数据集 —— #
    sizes = [4950, 1986,]  # 各子数据集样本数
    filename = ['/data01/gxs/dataset/for_predict/spec/test.h5',
                '/data01/gxs/dataset/for_predict/kol/test.h5',
                ]    
    batch_sizes = [3, 2,]
    
    datasets = [InitDataset_sample(filename[idx], sz) for idx, sz in enumerate(sizes)]

    # —— 3. 为每个子数据集创建 DataLoader —— #
    loaders = [
        DataLoader(
            dataset=ds,
            batch_size=bs,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        for ds, bs in zip(datasets, batch_sizes)
    ]
    print(f"Batch sizes for each dataset: {batch_sizes}")

    # ---------- 简单示例模型 ----------
    class Simple3DNet(torch.nn.Module):
        def __init__(self, in_ch=2, out_ch=2):  # 假设 u,f,c,re 各 2 通道，共 8
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
            )
        def forward(self, u, f, c, re):
            u = u[:,:,1:,:,:]
            u_sha = u.shape
            u = torch.transpose(u, 1, 2).reshape(-1, u_sha[1], u_sha[3], u_sha[4])
            out =  self.net(u)
            out = torch.reshape(out, (u_sha[0], 31, 2, 32, 32)).transpose(1,2)
                    
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---------- 训练循环 ----------
    num_epochs = 2

    for epoch in range(1, num_epochs + 1):
        model.train()
        step = 0
        epoch_loss = 0.0
        
        # 并行遍历到最长 DataLoader 耗尽
        # 统计总steps数（用于tqdm进度条）
        total_steps = max(len(loader) for loader in loaders)
        pbar = tqdm(zip_longest(*loaders, fillvalue=None), total=total_steps, desc=f"Epoch {epoch:02d}")
        
        for batches in pbar:
            batch_list = [b for b in batches if b is not None]
            if not batch_list:   # 所有 loader 都空，跳出
                break

            # --- 把 6 个 batch 拆解并记录长度 ---
            us, fs, cs, res, u_gts = zip(*batch_list)   # 每个元素是 Tensor [B_i, ...]
            batch_sizes = [u.shape[0] for u in us]

            # 拼接
            u_all   = torch.cat(us,   dim=0).to(device)   # [ΣB, C, T, H, W]
            f_all   = torch.cat(fs,   dim=0).to(device)
            c_all   = torch.cat(cs,   dim=0).to(device)
            re_all  = torch.cat(res,  dim=0).to(device)
            gt_all  = torch.cat(u_gts,dim=0).to(device)
            gt_all = gt_all[:, :,1:,:,:]

            # 前向
            pred_all = model(u_all, f_all, c_all, re_all)

            # --- 分段计算损失 ---
            start = 0
            loss_total = []
            for bs in batch_sizes:
                end = start + bs
                
                # reduction='none' 保留元素维度，然后在 (C,T,H,W) 上求均值
                import torch.nn.functional as F
                per_sample_mse = F.mse_loss(
                    pred_all[start:end],           # [bs, C, T, H, W]
                    gt_all[start:end],
                    reduction='none'
                ).reshape(bs, -1).mean(dim=1)         # [bs]  ——  每个样本的 MSE
                seg_loss = per_sample_mse.mean()   # 对本段 batch 再做平均
                
                # # ∑(pred-gt)^2 / bs   （不除通道/体素数）
                # seg_loss = torch.sum((pred_all[start:end] - gt_all[start:end]) ** 2) / bs            
                
                loss_total.append(seg_loss) 
                start = end

            # 对 6 段再取平均
            loss_total = torch.stack(loss_total).mean()

            # 反向 & 更新
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item()
            step += 1
            
            pbar.set_postfix({'avg_loss': f'{epoch_loss / step:.6f}'})

        print(f"Epoch {epoch:02d} | steps {step:4d} | avg_loss {epoch_loss / step:.6f}")