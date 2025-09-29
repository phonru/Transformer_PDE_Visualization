import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.autograd import Variable
# DEVICE = 'cuda:0'
import numpy as np
import time


def clones(module, N):
    """产生 N 个相同的层。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续encode N次(这里为6次)
        这里的Encoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """ 构造层范数模块（详见引文）."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory
        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)
    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = scores.cpu()
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        # 兼容 float16，避免溢出
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, min_value)
    # 将mask后的attention矩阵按照最后一个维度进行softmax，归一化到0~1
    p_attn = F.softmax(scores, dim=-1)
    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    # p_attn = p_attn.cuda()
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # h为head数量，保证可以整除，论文中该值是8
        assert d_model % h == 0
        # 得到一个head的attention表示维度，论文中是512/8=64
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, device='cuda:0'):  # max_len=256
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=device)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len, device=device).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号），已经忘记中学对数操作的同学请自行补课哈
        div_term = torch.exp(torch.arange(0., d_model, 2, device=device) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)



def subsequent_mask(size):
    " 掩盖后续 positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# ---------- 工具函数 ----------
def add_spatiotemporal_markers(u, time_embedding=True):                      # u: (B,2,T,H,W)=float
    B,_,T,H,W = u.shape
    x = torch.linspace(0,1,W,device=u.device).view(1,1,1,1,W).expand(B,1,T,H,W)
    y = torch.linspace(0,1,H,device=u.device).view(1,1,1,H,1).expand(B,1,T,H,W)
    t = torch.linspace(0,1,T,device=u.device).view(1,1,T,1,1).expand(B,1,T,H,W)
    if time_embedding:
        return torch.cat([u,x,y,t],dim=1)                   # (B,5,T,H,W)
    else:
        return torch.cat([u,x,y],dim=1)    # (B,4,T,H,W)


# class BlockPatchEmbed(nn.Module):
#     """把 32×32 的 block 压成单个 token"""
#     def __init__(self, in_c=5, embed_dim=2048):
#         super().__init__()
#         self.proj = nn.Sequential(                      # 32→1
#             nn.Conv2d(in_c, embed_dim//4, 4, 4), nn.GELU(),  # 32→8
#             nn.Conv2d(embed_dim//4, embed_dim, 8, 8),   # 8→1
#             # nn.AdaptiveAvgPool2d(1)                     # (B,E,1,1)
#         )
#     def forward(self, x):                               # x: (B,5,32,32)
#         return self.proj(x).flatten(2).transpose(1,2)   # (B,1,E)

class BlockPatchEmbed(nn.Module):
    """使用单个卷积层将图像块直接编码为Token，这是ViT和FNO中的标准做法"""
    def __init__(self, patch_size=32, in_c=5, embed_dim=2048):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (B, E, 1, 1)
        return x.flatten(2).transpose(1, 2) # (B, 1, E)
    
##################################################
class PatchEmbed2D(nn.Module):   # input (b,c,h,w,t)
    """
    对3D图像作Patch Embedding操作
    """
    def __init__(self, img_size=320, patch_size=80, in_c=2, embed_dim=2048, norm_layer=None):
        """
        此函数用于初始化相关参数
        :param img_size: 输入图像的大小
        :param patch_size: 一个patch的大小
        :param in_c: 输入图像的通道数
        :param embed_dim: 输出的每个token的维度
        :param norm_layer: 指定归一化方式，默认为None
        """
        super(PatchEmbed2D, self).__init__()
        img_size = (img_size, img_size)  # 224 -> (224, 224)
        patch_size = (patch_size, patch_size)  # 16 -> (16, 16)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 计算原始图像被划分为(14, 14)个小块
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 计算patch的个数为14*14=196个
        # 定义卷积层
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        # 定义归一化方式
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        此函数用于前向传播
        :param x: 原始图像
        :return: 处理后的图像
        """
        # 对图像依次作卷积、展平和调换处理: [B, C, H, W, T] -> [B, C, HWT] -> [B, HWT, C]
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        # 归一化处理
        x = self.norm(x)
        return x


class PatchUnembedding(nn.Module):
    """
    把 N_token (=patch_dim²) × embed_dim 还原为
    (B, 2, patch_dim_y, patch_img_size, patch_dim_x, patch_img_size)
       = (B, 2, m, 32, m, 32) when N_token=pn
    """
    def __init__(self, patch_img_size=32, in_channels=2, embed_dim=128,
                 out_format='6d'):
        """
        out_format:
          '6d' → (B,2,m,32,m,32)   —— 便于后续自定义拼接/平滑
          'chw'→ (B,2,320,320)       —— 若想一步得到整幅图像
        """
        super().__init__()
        self.patch_img_size  = patch_img_size
        self.in_channels = in_channels
        self.embed_dim   = embed_dim
        self.out_format  = out_format.lower()
        self.linear      = nn.Linear(embed_dim,
                                     in_channels * patch_img_size * patch_img_size)

    def forward(self, x):                       # x: (B, N_token, embed_dim)
        B, N, _ = x.shape
        # patch_dim = sqrt(patch_num)

        patch_dim = int(round(N ** 0.5))        
        assert patch_dim * patch_dim == N, \
            f"N_token 必须是平方数，收到 {N}"
        # (1) token → 像素块
        x = self.linear(x)                      # (B, N, 2*32*32)
        x = x.view(B, patch_dim, patch_dim,
                   self.in_channels,
                   self.patch_img_size, self.patch_img_size)         # (B,m,m,2,32,32)
        # (2) 调整维度顺序
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()          # (B,2,m,32,m,32)

        if self.out_format == 'chw':           # 可选一步拼成整幅场
            H = patch_dim * self.patch_img_size    # =320
            W = H
            x = x.view(B, self.in_channels, H, W)             # (B,2,320,320)
        return x



def draw_add_embedding():
    import matplotlib.pyplot as plt
    import torch

    # 假设 B, T, H, W, u 已定义
    # 例如：B, T, H, W = 2, 100, 256, 256; u = torch.randn(B,2,T,H,W,device='cuda')
    B, T, H, W = 2, 100, 320, 320
    u = torch.randn(B,2,T,H,W,device='cuda')

    # 生成编码
    x = torch.linspace(0,1,W,device=u.device).view(1,1,1,1,W).expand(B,1,T,H,W)
    y = torch.linspace(0,1,H,device=u.device).view(1,1,1,H,1).expand(B,1,T,H,W)
    t = torch.linspace(0,1,T,device=u.device).view(1,1,T,1,1).expand(B,1,T,H,W)

    # 取 batch=0, t=0
    x_img = x[0,0,0].cpu().numpy()  # (H, W)
    y_img = y[0,0,0].cpu().numpy()  # (H, W)
    t_img = t[0,0,0].cpu().numpy()  # (H, W)

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(x_img, cmap='viridis')
    plt.title('X Channel')
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(y_img, cmap='viridis')
    plt.title('Y Channel')
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(t_img, cmap='viridis')
    plt.title('T Channel')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('./draw/spatiotemporal_encoding.png')
    plt.close()



############################################################################
#                                                                          #
# ======================== 模型改进部分 (V4) ============================== #
#            (CNN Stem + Spatiotemporal Factorized Attention)              #
#                                                                          #
############################################################################

class ResBlock(nn.Module):
    """一个标准的残差卷积块，用于构建CNN Stem"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# class CNNStem(nn.Module):
#     """使用CNN提取空间特征，并最终输出Patch Tokens"""
#     def __init__(self, in_c, base_dim, embed_dim, patch_size):
#         super().__init__()
#         # 初始卷积层
#         self.conv_in = nn.Conv2d(in_c, base_dim, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn_in = nn.BatchNorm2d(base_dim)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 几个残差块
#         self.res_blocks = nn.Sequential(
#             ResBlock(base_dim, base_dim * 2, stride=2), # H/2, W/2
#             ResBlock(base_dim * 2, base_dim * 4, stride=2), # H/4, W/4
#         )
#
#         # 最终的Patchify卷积层
#         # 输入维度是 base_dim * 4, 输出维度是 embed_dim
#         # Kernel size 和 stride 保证了将特征图转换为不重叠的patch token
#         final_patch_size = patch_size // 4
#         self.patchify = nn.Conv2d(base_dim * 4, embed_dim,
#                                   kernel_size=final_patch_size,
#                                   stride=final_patch_size)
#
#     def forward(self, x):
#         x = self.relu(self.bn_in(self.conv_in(x)))
#         x = self.res_blocks(x)
#         x = self.patchify(x) # (B, E, H', W')
#         return x.flatten(2).transpose(1, 2) # (B, pn, E)
class CNNEmbed(nn.Module):
    """
    将图像分割成补丁（Patch），并通过一个卷积层进行线性嵌入。

    Input:
        x: (B, C, H, W) 输入张量
    Output:
        (B, num_patches, embed_dim) 输出张量
    """

    def __init__(self, in_c, conv_dim, embed_dim, patch_size):
        """
        参数:
            in_c (int): 输入图像的通道数 (例如，RGB图像为3)。
            embed_dim (int): 每个补丁嵌入后的目标维度。
            patch_size (int): 每个正方形补丁的边长。
        """
        super().__init__()
        self.patch_size = patch_size
        # 使用一个卷积层实现补丁划分和特征嵌入
        # kernel_size 和 stride 相等，确保补丁之间不重叠
        self.proj = nn.Conv2d(
            in_c,
            conv_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.linear1 = nn.Linear(conv_dim, conv_dim)
        self.linear2 = nn.Linear(conv_dim, embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x 的形状: (B, C, H, W)
        """
        # 1. 通过卷积进行投影
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)

        # 2. 展平空间维度并调整顺序以匹配Transformer输入格式
        # (B, E, H', W') -> (B, E, N) -> (B, N, E)
        # 其中 N = H' * W' 是补丁的数量
        x = x.flatten(2).transpose(1, 2)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SpatiotemporalEncoderLayer(nn.Module):
    """
    分解式时空注意力编码器层。
    在一个Block内，依次进行时间注意力和空间注意力计算。
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.temporal_attn = copy.deepcopy(self_attn)
        self.spatial_attn = copy.deepcopy(self_attn)
        self.feed_forward = feed_forward
        # 3个子层连接：时间注意力，空间注意力，前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, temporal_mask):
        """
        Args:
            x (torch.Tensor): 输入张量, shape (B, pn, T, D)
            temporal_mask (torch.Tensor): 时间上的因果掩码
        """
        B, pn, T, D = x.shape
        
        # 1. 时间注意力
        # Reshape: (B, pn, T, D) -> (B*pn, T, D)
        x_temp = x.reshape(B * pn, T, D)
        x_temp = self.sublayer[0](x_temp, lambda q: self.temporal_attn(q, q, q, temporal_mask))
        x = x_temp.view(B, pn, T, D)
        
        # 2. 空间注意力
        # Reshape: (B, pn, T, D) -> (B, T, pn, D) -> (B*T, pn, D)
        x_spatial = x.permute(0, 2, 1, 3).reshape(B * T, pn, D)
        # 空间上不需要mask
        x_spatial = self.sublayer[1](x_spatial, lambda q: self.spatial_attn(q, q, q, None)) 
        x = x_spatial.view(B, T, pn, D).permute(0, 2, 1, 3)
        
        # 3. 前馈网络
        x = self.sublayer[2](x, self.feed_forward)
        return x

class SpatiotemporalEncoder(nn.Module):
    """时空编码器，由N个SpatiotemporalEncoderLayer堆叠而成"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class PDE_CNN_Spatiotemporal_Transformer(nn.Module):
    """
    最终集成的模型：
    1. 使用CNN Stem进行空间特征提取。
    2. 使用分解式时空注意力编码器进行演化建模。
    """
    def __init__(self, d_model=512, nhead=8, num_st_layers=6,
                 d_ff=2048, dropout=0.1, img_size=160, patch_size=32,
                 cnn_base_dim=64, device='cuda:0'):
        super().__init__()
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size
        self.num_patches = self.num_patches_side ** 2
        self.device = device

        # 1. CNN Stem 空间编码器
        # self.u0_cnn_stem = CNNStem(in_c=4, base_dim=cnn_base_dim, embed_dim=d_model, patch_size=patch_size)
        # self.ft_cnn_stem = CNNStem(in_c=5, base_dim=cnn_base_dim, embed_dim=d_model, patch_size=patch_size)

        self.u0_cnn_stem = CNNEmbed(in_c=4, conv_dim=d_model, embed_dim=d_model, patch_size=patch_size)
        self.ft_cnn_stem = CNNEmbed(in_c=5, conv_dim=d_model, embed_dim=d_model, patch_size=patch_size)

        # 2. 边界条件与参数嵌入
        self.bc_embed = nn.Sequential(
            nn.Linear(8 * img_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.re_embed = nn.Linear(1, d_model)
        
        # 3. 特征融合层  
        # --- 修改前 (加法融合) ---
        # 将 u0, ft, bc, re 四种token相加融合
        # self.fusion_layer_norm = LayerNorm(d_model)

        # --- 修改后 (拼接融合) ✅ ---
        fusion_input_dim = d_model * 4  # u0, ft, bc, re 四个特征拼接
        self.fusion_layer = nn.Linear(fusion_input_dim, d_model)

        # 4. 时空位置编码
        self.temporal_pos_encoder = PositionalEncoding(d_model, dropout, device=device)
        self.spatial_pos_encoder = nn.Parameter(torch.zeros(1, self.num_patches, 1, d_model))
        nn.init.trunc_normal_(self.spatial_pos_encoder, std=.02)

        # 5. 分解式时空演化编码器
        st_layer = SpatiotemporalEncoderLayer(d_model, 
                                            MultiHeadedAttention(nhead, d_model, dropout),
                                            PositionwiseFeedForward(d_model, d_ff, dropout), 
                                            dropout)
        self.st_encoder = SpatiotemporalEncoder(st_layer, num_st_layers)
        
        # 6. 输出头
        self.unpatcher = PatchUnembedding(
            patch_img_size=patch_size, in_channels=2,
            embed_dim=d_model, out_format='chw'
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, u0, ft_series, bc_series, re_series):
        B, T, _, H, W = ft_series.shape
        
        # === 步骤 1: CNN Stem 提取空间特征 ===
        u0_with_coords = add_spatiotemporal_markers(u0.unsqueeze(2), time_embedding=False).squeeze(2) # (B, 4, H, W)
        u0_tokens = self.u0_cnn_stem(u0_with_coords).unsqueeze(1).expand(-1, T, -1, -1) # (B, T, pn, D)
        
        ft_with_coords = add_spatiotemporal_markers(ft_series.permute(0, 2, 1, 3, 4)) # (B,5,T,H,W)
        ft_with_coords = ft_with_coords.permute(0, 2, 1, 3, 4).reshape(B * T, 5, H, W) # (B*T, 5, H, W)
        ft_tokens = self.ft_cnn_stem(ft_with_coords).view(B, T, self.num_patches, -1) # (B, T, pn, D)
        
        # === 步骤 2: 嵌入条件信息并融合 ===
        bc_flat = bc_series.reshape(B * T, -1)
        bc_tokens = self.bc_embed(bc_flat).view(B, T, 1, -1).expand(-1, -1, self.num_patches, -1) # (B, T, pn, D)
        
        re_tokens = self.re_embed(re_series.unsqueeze(-1)).view(B, T, 1, -1).expand(-1, -1, self.num_patches, -1) # (B, T, pn, D)
        

        # --- 修改前 (加法融合) ---
        # # 使用加法融合, 更灵活
        # fused_tokens = u0_tokens + ft_tokens + bc_tokens + re_tokens
        # fused_tokens = self.fusion_layer_norm(fused_tokens) # (B, T, pn, D)
        # --- 修改后 (拼接融合) ✅ ---
        # 拼接所有特征
        combined_features = torch.cat([u0_tokens, ft_tokens, bc_tokens, re_tokens], dim=-1) # (B, T, pn, 4*D)
        # 通过线性层进行融合
        fused_tokens = self.fusion_layer(combined_features) # (B, T, pn, D)

        # === 步骤 3: 添加时空位置编码 ===
        # (B, T, pn, D) -> (B, pn, T, D) 以匹配Encoder输入
        fused_tokens = fused_tokens.permute(0, 2, 1, 3)
        # 添加空间PE
        fused_tokens += self.spatial_pos_encoder
        # Reshape & 添加时间PE
        fused_tokens = fused_tokens.reshape(B * self.num_patches, T, self.d_model)
        fused_tokens = self.temporal_pos_encoder(fused_tokens)
        fused_tokens = fused_tokens.view(B, self.num_patches, T, self.d_model)

        # === 步骤 4: 时空演化建模 ===
        causal_mask = subsequent_mask(T).to(self.device)
        st_output = self.st_encoder(fused_tokens, causal_mask) # (B, pn, T, D)
        
        # === 步骤 5: 重构输出 ===
        # (B, pn, T, D) -> (B, T, pn, D)
        output_tokens = st_output.permute(0, 2, 1, 3).contiguous()
        unpatch_input = output_tokens.reshape(B * T, self.num_patches, self.d_model)
        
        predicted_fields_flat = self.unpatcher(unpatch_input) # (B*T, 2, H, W)
        predicted_series = predicted_fields_flat.view(B, T, 2, H, W)
        
        return predicted_series




if __name__ == "__main__":
    # 切换到新模型进行测试
    print("Testing the new PDECausalEncoderModel model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 定义模型参数
    B, T, H, W = 2, 16, 160, 160 # 减小 Batch Size 和 Time-steps 以便快速测试
    d_model = 1024              # 减小模型尺寸以适应本地测试
    patch_size = 32
    spatial_embed_dim = 256
    nhead = 8
    
    # 准备伪数据
    u0 = torch.randn(B, 2, H, W, device=device)
    ft = torch.randn(B, T-1, 2, H, W, device=device)
    bc = torch.randn(B, T-1, 8, W, device=device)
    re = torch.randn(B, T-1, device=device)
    
    # 实例化新模型
    model_v4 = PDE_CNN_Spatiotemporal_Transformer(
        d_model=d_model,
        nhead=nhead,
        img_size=H, 
        patch_size=patch_size,
        device=device
    ).to(device)
    
    # 打印模型参数量
    param_count = sum(p.numel() for p in model_v4.parameters() if p.requires_grad)
    print(f"Model V3 trainable parameters: {param_count / 1e6:.2f}M")
    
    # 前向传播
    try:
        # 新模型一次性输出所有时刻的解
        start_time = time.time()
        output_series = model_v4(u0, ft, bc, re)
        end_time = time.time()
        
        # 检查输出形状
        print(f"\nInput u0 shape: {u0.shape}")
        print(f"Input ft_series shape: {ft.shape}")
        print(f"Predicted series shape: {output_series.shape}")
        
        expected_shape = (B, T-1, 2, H, W)
        assert output_series.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output_series.shape}"
        
        print(f"\nForward pass took {end_time - start_time:.4f} seconds.")
        print("✅ Model V4 forward pass successful!")
        print("✅ Architecture refactoring based on your idea is complete and seems correct.")

    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")
        import traceback
        traceback.print_exc()