import math

import torch
from einops import rearrange
from timm.models.layers import DropPath,trunc_normal_
from torch import nn,einsum
from torch.nn import functional as F
from typing import Tuple
from einops.layers.torch import Rearrange,Reduce


class BA(nn.Module):
    """
    BA模块：边界注意力模块（Boundary Attention）
    
    使用深度可分离卷积和条带卷积提取多尺度边界特征
    结合5x1、1x5、3x1、1x3等多种卷积核捕捉不同方向的边界信息
    """
    
    def __init__(self, in_channels):
        """
        初始化BA模块
        
        Args:
            in_channels (int): 输入通道数
        """
        super(BA, self).__init__()
        self.depthwise_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.strip_conv_5x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0),
                                        groups=in_channels)
        self.strip_conv_1x5 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2),
                                        groups=in_channels)
        self.strip_conv_1x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1),
                                        groups=in_channels)
        self.strip_conv_3x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0),
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播：提取边界注意力特征
        
        Args:
            x (torch.Tensor): 输入特征图
            
        Returns:
            torch.Tensor: 增强后的特征图
        """
        device = x.device
        self.depthwise_conv1 = self.depthwise_conv1.to(device)
        self.strip_conv_5x1 = self.strip_conv_5x1.to(device)
        self.strip_conv_1x5 = self.strip_conv_1x5.to(device)
        self.strip_conv_1x3 = self.strip_conv_1x3.to(device)
        self.strip_conv_3x1 = self.strip_conv_3x1.to(device)
        self.pointwise_conv = self.pointwise_conv.to(device)
        self.bn = self.bn.to(device)
        self.gelu = self.gelu.to(device)
        self.sigmoid = self.sigmoid.to(device)

        x_DWConv = self.depthwise_conv1(x)

        x_DWStripConv5=self.sigmoid(self.strip_conv_1x5(self.gelu(self.strip_conv_5x1(x_DWConv))))
        x_DWStripConv3 = self.sigmoid(self.strip_conv_1x3(self.gelu(self.strip_conv_3x1(x_DWConv))))
        x_DWStripConv = x_DWStripConv5 + x_DWStripConv3

        x_hadamard = x * x_DWStripConv
        x_fusion = x_hadamard + x

        x_Conv_Stage1 = self.gelu(self.bn(self.pointwise_conv(x_fusion)))
        x_Conv_Stage2 = self.gelu(self.bn(self.pointwise_conv(x_Conv_Stage1)))


        return x_Conv_Stage2


def bacam_op(features, ghost_mul, ghost_add, h_attn, lam, gamma,
            kernel_size=5, dilation=1, stride=1, version=''):
    """
    BACAM操作函数：边界注意力卷积注意力模块的核心运算
    
    结合ghost卷积和注意力机制，对特征图进行增强
    
    Args:
        features (torch.Tensor): 输入特征图
        ghost_mul (torch.Tensor): ghost乘法参数
        ghost_add (torch.Tensor): ghost加法参数
        h_attn (torch.Tensor): 注意力图
        lam (float): lambda参数，控制ghost_mul的影响
        gamma (float): gamma参数，控制ghost_add的影响
        kernel_size (int): 卷积核大小
        dilation (int): 膨胀率
        stride (int): 步长
        version (str): 版本标识
        
    Returns:
        torch.Tensor: 增强后的特征图
    """
    _B, _C = features.shape[:2]
    ks = kernel_size
    ghost_mul = ghost_mul ** lam if lam != 0 \
        else torch.ones(_B, _C, ks, ks, device=features.device, requires_grad=False)
    ghost_add = ghost_add * gamma if gamma != 0 \
        else torch.zeros(_B, _C, ks, ks, device=features.device, requires_grad=False)
    B, C, H, W = features.shape
    _pad = kernel_size // 2 * dilation

    ba = BA(C)
    features = ba(features)

    features = F.unfold(
        features, kernel_size=kernel_size, dilation=dilation, padding=_pad, stride=stride) \
        .reshape(B, C, kernel_size ** 2, H * W)
    ghost_mul = ghost_mul.reshape(B, C, kernel_size ** 2, 1)
    ghost_add = ghost_add.reshape(B, C, kernel_size ** 2, 1)
    h_attn = h_attn.reshape(B, 1, kernel_size ** 2, H * W)
    filters = ghost_mul * h_attn + ghost_add
    return (features * filters).sum(2).reshape(B, C, H, W)


class BACAM(nn.Module):
    """
    BACAM类：增强的局部自注意力模块
    
    结合了边界注意力、ghost卷积和多头自注意力机制
    用于提取更丰富的空间特征
    """

    def __init__(self, dim, num_heads, kernel_size=5,
                 stride=1, dilation=1, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., group_width=8, groups=1, lam=1,
                 gamma=1, **kwargs):
        """
        初始化BACAM模块
        
        Args:
            dim (int): 特征维度
            num_heads (int): 注意力头数
            kernel_size (int): 卷积核大小
            stride (int): 步长
            dilation (int): 膨胀率
            qkv_bias (bool): QKV是否使用偏置
            qk_scale (float, optional): QK缩放因子
            attn_drop (float): 注意力dropout率
            proj_drop (float): 投影dropout率
            group_width (int): 分组宽度
            groups (int): 分组数
            lam (float): lambda参数
            gamma (float): gamma参数
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dim_qk = self.dim // 3 * 2
        self.dim_v = dim



        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        head_dim = self.dim_v // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.dim_qk % group_width != 0:
            self.dim_qk = math.ceil(float(self.dim_qk) / group_width) * group_width

        self.group_width = group_width
        self.groups = groups
        self.lam = lam
        self.gamma = gamma
        print(f'lambda = {lam}, gamma = {gamma}, scale = {self.scale}')

        self.pre_proj = nn.Conv2d(dim, self.dim_qk * 2 + self.dim_v, 1, bias=qkv_bias)
        self.attn = nn.Sequential(
            nn.Conv2d(self.dim_qk, self.dim_qk, kernel_size, padding=(kernel_size // 2) * dilation,
                      dilation=dilation, groups=self.dim_qk // group_width),
            nn.GELU(),
            nn.Conv2d(self.dim_qk, kernel_size ** 2 * num_heads, 1, groups=groups))

        if self.lam != 0 and self.gamma != 0:
            ghost_mul = torch.randn(1, 1, self.dim_v, kernel_size, kernel_size)
            ghost_add = torch.zeros(1, 1, self.dim_v, kernel_size, kernel_size)
            trunc_normal_(ghost_add, std=.02)
            self.ghost_head = nn.Parameter(torch.cat((ghost_mul, ghost_add), dim=0), requires_grad=True)
        elif self.lam == 0 and self.gamma != 0:
            ghost_add = torch.zeros(1, self.dim_v, kernel_size, kernel_size)
            trunc_normal_(ghost_add, std=.02)
            self.ghost_head = nn.Parameter(ghost_add, requires_grad=True)
        elif self.lam != 0 and self.gamma == 0:
            ghost_mul = torch.randn(1, self.dim_v, kernel_size, kernel_size)
            self.ghost_head = nn.Parameter(ghost_mul, requires_grad=True)
        else:
            self.ghost_head = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.post_proj = nn.Linear(self.dim_v, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv_q = nn.Conv2d(dim, dim, 3, 1, "same", bias=True,
                                groups=dim)
        self.layernorm_q = nn.LayerNorm(dim, eps=1e-5)
        self.conv_k = nn.Conv2d(dim, dim, 3, 1, "same", bias=True,
                                groups=dim)
        self.layernorm_k = nn.LayerNorm(dim, eps=1e-5)
        self.conv_v = nn.Conv2d(dim, dim, 3, 1, "same", bias=True,
                                groups=dim)
        self.layernorm_v = nn.LayerNorm(dim, eps=1e-5)

        self.conv_qk = nn.Conv2d(dim, self.dim_qk, 1)

    def _build_projection(self, x, qkv):
        """
        构建Q/K/V投影
        
        Args:
            x (torch.Tensor): 输入特征
            qkv (str): 投影类型（"q"、"k" 或 "v"）
            
        Returns:
            torch.Tensor: 投影后的特征
        """

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        """
        执行Q/K/V的卷积投影
        
        Args:
            x (torch.Tensor): 输入特征图
            
        Returns:
            tuple: (q, k, v) 查询、键、值三元组
        """
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")
        return q, k, v


    def forward(self, x, H, W, mask=None):
        """
        前向传播：执行BACAM注意力计算
        
        Args:
            x (torch.Tensor): 输入序列 [B, N, C]
            H (int): 特征图高度
            W (int): 特征图宽度
            mask (optional): 注意力掩码
            
        Returns:
            torch.Tensor: 输出特征序列
        """
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        C = self.dim_v
        ks = self.kernel_size
        G = self.num_heads
        x = x.permute(0, 3, 1, 2)


        q, k, v = self.forward_conv(x)
        q = self.conv_qk(q)
        k = self.conv_qk(k)

        hadamard_product = q * k * self.scale

        if self.stride > 1:
            hadamard_product = F.avg_pool2d(hadamard_product, self.stride)

        h_attn = self.attn(hadamard_product)

        v = v.reshape(B * G, C // G, H, W)
        h_attn = h_attn.reshape(B * G, -1, H, W).softmax(1)
        h_attn = self.attn_drop(h_attn)

        ghost_mul = None
        ghost_add = None
        if self.lam != 0 and self.gamma != 0:
            gh = self.ghost_head.expand(2, B, C, ks, ks).reshape(2, B * G, C // G, ks, ks)
            ghost_mul, ghost_add = gh[0], gh[1]
        elif self.lam == 0 and self.gamma != 0:
            ghost_add = self.ghost_head.expand(B, C, ks, ks).reshape(B * G, C // G, ks, ks)
        elif self.lam != 0 and self.gamma == 0:
            ghost_mul = self.ghost_head.expand(B, C, ks, ks).reshape(B * G, C // G, ks, ks)

        x = bacam_op(v, ghost_mul, ghost_add, h_attn, self.lam, self.gamma,
                    self.kernel_size, self.dilation, self.stride)
        x = x.reshape(B, C, H // self.stride, W // self.stride)
        x = self.post_proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        x = x.reshape(B, N, C)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )


        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

        self.conv_q = nn.Conv2d(dim, dim, 3, 1, "same", bias=True,
                                groups=dim)
        self.layernorm_q = nn.LayerNorm(dim, eps=1e-5)
        self.conv_k = nn.Conv2d(dim, dim, 3, 1, "same", bias=True,
                                groups=dim)
        self.layernorm_k = nn.LayerNorm(dim, eps=1e-5)
        self.conv_v = nn.Conv2d(dim, dim, 3, 1, "same", bias=True,
                                groups=dim)
        self.layernorm_v = nn.LayerNorm(dim, eps=1e-5)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v



    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        B,N,C = x.shape
        x = x.permute(0,2,1)
        x = x.reshape(B,C,window_height,window_height)
        q, k, v = self.forward_conv(x)
        q = q.reshape(B,C,window_height*window_height)
        q = q.permute(0,2,1)
        k = k.reshape(B, C, window_height * window_height)
        k = k.permute(0, 2, 1)
        v = v.reshape(B, C, window_height * window_height)
        v = v.permute(0, 2, 1)



        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),

        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                                kernel_size//2, groups=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x1 = x
        x = self.dwconv(x)
        x = x + x1
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HAM(nn.Module):
    def __init__(self, layer_dim, window_size = 7, dim_head = 32,
        dropout = 0.1,mbconv_shrinkage_rate = 0.25, mbconv_expansion_rate = 4):
        super().__init__()

        self.conv = MBConv(
            layer_dim,
            layer_dim,
            downsample=False,
            expansion_rate=mbconv_expansion_rate,
            shrinkage_rate=mbconv_shrinkage_rate
        )
        self.attn = BACAM(layer_dim, dim_head)
        self.window_size = window_size
        w = window_size
        self.block_attn = Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)
        self.grid_attn = Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)

        self.proj = nn.Conv2d(2*layer_dim, layer_dim, 1, 1, 0, bias=True)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(dropout)

        self.ity = nn.Identity()

        self.mlp = ConvFFN(in_channels=layer_dim, hidden_channels=layer_dim * 4, kernel_size=5, stride=1,
                           out_channels=layer_dim, drop_out=dropout)

    def forward(self, x: torch.Tensor,H,W):
        B,N,C= x.shape
        local_x = x
        x = x.permute(0,2,1)
        x = x.reshape(B,C,H,W)
        res = []
        x = self.conv(x)
        w = self.window_size
        global_x = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)(x)
        global_x = self.grid_attn(global_x)
        global_x = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')(global_x)
        res.append(global_x)

        local_x = self.attn(local_x,H,W)
        local_x = local_x.permute(0,2,1)
        local_x = local_x.reshape(B,C,H,W)
        res.append(local_x)

        out = self.proj_drop(self.proj(torch.cat(res, dim=1)))

        out = x + self.drop_path(out)
        out = self.ity(x) + self.drop_path(self.mlp(out))

        out= out.permute(0,2,3,1)
        out = out.reshape(B,N,C)

        return out





class SelfAtten(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1,
                                                                       3)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if (len(self.reduction_ratio) == 4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        elif (len(self.reduction_ratio) == 3):
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if (len(self.reduction_ratio) == 4):
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if (len(self.reduction_ratio) == 3):
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out




class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode="add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class MIS(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HAM(dim,dim_head=head)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx




class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class MyEncoder(nn.Module):
    def __init__(self, image_size, dims, layers, token_mlp='mix_skip'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3,
                                                   dims[0])  # Embeding的作用是使2D图片成为适应trasformer的数据类型
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        # transformer encoder
        self.block1 = nn.ModuleList([
            MIS(dims[0], heads[0], reduction_ratios[0], token_mlp)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            MIS(dims[1], heads[1], reduction_ratios[1], token_mlp)
            for _ in range(layers[1])])


        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            MIS(dims[2], heads[2], reduction_ratios[2], token_mlp)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            MIS(dims[3], heads[3], reduction_ratios[3], token_mlp)
            for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs





class Decoder(nn.Module):
    def __init__(self, dims, embed_dim, num_classes):
        super().__init__()

        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

        self.conv_seg = nn.Conv2d(128, num_classes, 1)

        self.dropout = nn.Dropout2d(
            0.1)


    def forward(self, inputs: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[
            3])

        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear',
                            align_corners=False)
        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)


segformer_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2], 256],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2], 256],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3], 768],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3], 768],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3], 768],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3], 768]
}


class SegFormer(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in segformer_settings.keys(), f"SegFormer model name should be in {list(segformer_settings.keys())}"
        dims, layers, embed_dim = segformer_settings[model_name]

        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'),
                                          strict=False)


        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(
                        m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m,
                                nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)

