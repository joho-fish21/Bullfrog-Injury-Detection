import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================
# 6种注意力机制（输入输出通道一致，均为实际值）
# ==============================================================
class SEAttention(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        mid = max(8, c1 // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ECAAttention(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, c1, c2, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(8, c1 // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, c1, 1, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        x = x * self.sigmoid(avg_out + max_out)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))


class CoordinateAttention(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super().__init__()
        mid = max(8, c1 // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mid, c1, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.sigmoid(self.conv_h(x_h)) * self.sigmoid(self.conv_w(x_w))


class EMA(nn.Module):
   
    def __init__(self, c1, c2, factor=8):
        super().__init__()
       
        self.groups = factor
        while c1 % self.groups != 0 and self.groups > 1:
            self.groups -= 1
        self.ch_per_group = c1 // self.groups
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_ch = max(1, self.ch_per_group // 4)
        self.attention = nn.Sequential(
            nn.Conv2d(self.ch_per_group, mid_ch, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_ch, self.ch_per_group, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
    
        factor = self.groups
        ch_per_group = self.ch_per_group
        
        x_grouped = x.view(B, factor, ch_per_group, H, W)
        x_flat = x_grouped.reshape(B * factor, ch_per_group, H, W)
        y = self.avg_pool(x_flat)
        y = self.attention(y)
        y = y.view(B, factor, ch_per_group, 1, 1)
        return (x_grouped * y).view(B, C, H, W)



class SimAM(nn.Module):
    def __init__(self, c1, c2, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.sigmoid(y)





class GGCA(nn.Module):
    
    def __init__(self, c1, c2, reduction=16, num_groups=4):
        super().__init__()
        self.num_groups = num_groups
        assert c1 % num_groups == 0, "通道数必须能被分组数整除"
        self.group_channels = c1 // num_groups
        
        mid_ch = max(1, self.group_channels // reduction)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.group_channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, self.group_channels, 1, bias=False),
        )
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_groups, self.group_channels, H, W)
        x_flat = x.reshape(B * self.num_groups, self.group_channels, H, W)
        
        # 方向全局池化
        x_h_avg = x_flat.mean(dim=3, keepdim=True)
        x_h_max, _ = x_flat.max(dim=3, keepdim=True)
        x_w_avg = x_flat.mean(dim=2, keepdim=True)
        x_w_max, _ = x_flat.max(dim=2, keepdim=True)
        
        # 共享卷积权重
        y_h_avg = self.shared_conv(x_h_avg)
        y_h_max = self.shared_conv(x_h_max)
        y_w_avg = self.shared_conv(x_w_avg)
        y_w_max = self.shared_conv(x_w_max)
        
        # 注意力加权
        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(B, self.num_groups, self.group_channels, H, 1)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(B, self.num_groups, self.group_channels, 1, W)
        
        return (x * att_h * att_w).view(B, C, H, W)



# 4种轻量化主干模块（输入输出均为实际通道值）
# ==============================================================

class PartialConv3(nn.Module):
    """"""
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv3(x1), x2), dim=1)


class FasterNetBlock(nn.Module):
   
    def __init__(self, c1, c2, *args, n_div=4, expand_ratio=1):
        super().__init__()
        assert c1 == c2, f"FasterNetBlock 为通道不变模块，要求 c1==c2，收到 {c1}->{c2}"
        dim = c2
        hidden = dim * expand_ratio
        self.pconv = PartialConv3(dim, n_div)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.mlp(self.pconv(x))


class InvertedResidual(nn.Module):
    
    def __init__(self, c1, c2, stride=1, expand_ratio=6):
        super().__init__()
        self.stride = stride
        hidden_dim = int(c1 * expand_ratio)
        self.use_res_connect = self.stride == 1 and c1 == c2
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)



# ==============================================================
class GhostConv(nn.Module):
    
    def __init__(self, c1, c2, ratio=2, dw_size=3, stride=1):
        super().__init__()
        init_ch = c2 // ratio
        cheap_ch = c2 - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(c1, init_ch, 1, stride, 0, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.SiLU()
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, dw_size, 1, dw_size // 2, groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch),
            nn.SiLU()
        )

    def forward(self, x):
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)


class GhostBottleneck(nn.Module):
    """"""
    def __init__(self, c1, c2, shortcut=True, e=2):
        super().__init__()
        hidden_dim = int(c1 * e)
        self.conv = nn.Sequential(
            GhostConv(c1, hidden_dim, stride=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            GhostConv(hidden_dim, c2, stride=1),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv(x) if self.add else self.conv(x)


class GhostBottleneckV2(GhostBottleneck):
    """"""


class C3Ghost(nn.Module):
   
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        hidden_dim = int(c2 * e)
        self.cv1 = GhostConv(c1, hidden_dim, stride=1)
        self.cv2 = GhostConv(c1, hidden_dim, stride=1)
        self.cv3 = GhostConv(2 * hidden_dim, c2, stride=1)
        self.m = nn.Sequential(*(GhostBottleneck(hidden_dim, hidden_dim, shortcut, e=2) for _ in range(n)))

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))



class ShuffleBottleneck(nn.Module):
    
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.stride = stride
        mid = c2 // 2
        if stride == 1:
            self.branch1 = nn.Identity()
            self.branch2 = nn.Sequential(
                nn.Conv2d(c1//2, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
                nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
                nn.Conv2d(mid, c2//2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2//2),
                nn.SiLU(),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(c1, c1, 3, stride, 1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2//2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2//2),
                nn.SiLU(),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(c1, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
                nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
                nn.Conv2d(mid, c2//2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2//2),
                nn.SiLU(),
            )
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class MBConv(nn.Module):
  
    def __init__(self, c1, c2, stride=1, expand_ratio=6, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        hidden_dim = int(c1 * expand_ratio)
        self.use_res_connect = stride == 1 and c1 == c2
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            SEAttention(hidden_dim, hidden_dim, reduction=int(1/se_ratio)),
            nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# ==============================================================
# 4种边界框损失函数
# ==============================================================
class GIouLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        p_x1,p_y1,p_x2,p_y2 = pred.chunk(4,-1)
        t_x1,t_y1,t_x2,t_y2 = target.chunk(4,-1)
        inter = (torch.min(p_x2,t_x2)-torch.max(p_x1,t_x1)).clamp(0) * (torch.min(p_y2,t_y2)-torch.max(p_y1,t_y1)).clamp(0)
        w1,h1 = p_x2-p_x1, p_y2-p_y1
        w2,h2 = t_x2-t_x1, t_y2-t_y1
        union = w1*h1 + w2*h2 - inter + 1e-7
        iou = inter / union
        cw,ch = torch.max(p_x2,t_x2)-torch.min(p_x1,t_x1), torch.max(p_y2,t_y2)-torch.min(p_y1,t_y1)
        giou = iou - (cw*ch - union) / (cw*ch + 1e-7)
        return 1 - giou

class SIouLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        p_x1,p_y1,p_x2,p_y2 = pred.chunk(4,-1)
        t_x1,t_y1,t_x2,t_y2 = target.chunk(4,-1)
        w1,h1 = p_x2-p_x1, p_y2-p_y1
        w2,h2 = t_x2-t_x1, t_y2-t_y1
        cx_p,cy_p = (p_x1+p_x2)/2, (p_y1+p_y2)/2
        cx_t,cy_t = (t_x1+t_x2)/2, (t_y1+t_y2)/2
        sigma = torch.sqrt((cx_t-cx_p)**2 + (cy_t-cy_p)**2)
        cw,ch = torch.max(p_x2,t_x2)-torch.min(p_x1,t_x1), torch.max(p_y2,t_y2)-torch.min(p_y1,t_y1)
        sin_alpha = torch.abs(cy_t-cy_p) / (sigma + 1e-7).clamp(0,1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha)*2 - math.pi/2)
        gamma = 2 - angle_cost
        dist_cost = 2 - torch.exp(-gamma*((cx_t-cx_p)/cw)**2) - torch.exp(-gamma*((cy_t-cy_p)/ch)**2)
        w_cost = torch.abs(w1-w2) / torch.max(w1,w2)
        h_cost = torch.abs(h1-h2) / torch.max(h1,h2)
        shape_cost = (1-torch.exp(-w_cost))**4 + (1-torch.exp(-h_cost))**4
        inter = (torch.min(p_x2,t_x2)-torch.max(p_x1,t_x1)).clamp(0) * (torch.min(p_y2,t_y2)-torch.max(p_y1,t_y1)).clamp(0)
        union = w1*h1 + w2*h2 - inter + 1e-7
        iou = inter / union
        return 1 - (iou - (dist_cost+shape_cost)/2)

class EIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        p_x1,p_y1,p_x2,p_y2 = pred.chunk(4,-1)
        t_x1,t_y1,t_x2,t_y2 = target.chunk(4,-1)
        w1,h1 = p_x2-p_x1, p_y2-p_y1
        w2,h2 = t_x2-t_x1, t_y2-t_y1
        cx_p,cy_p = (p_x1+p_x2)/2, (p_y1+p_y2)/2
        cx_t,cy_t = (t_x1+t_x2)/2, (t_y1+t_y2)/2
        rho2 = (cx_t-cx_p)**2 + (cy_t-cy_p)**2
        cw,ch = torch.max(p_x2,t_x2)-torch.min(p_x1,t_x1), torch.max(p_y2,t_y2)-torch.min(p_y1,t_y1)
        c2 = cw**2 + ch**2 + 1e-7
        w_loss = (w1-w2)**2 / (cw**2 + 1e-7)
        h_loss = (h1-h2)**2 / (ch**2 + 1e-7)
        inter = (torch.min(p_x2,t_x2)-torch.max(p_x1,t_x1)).clamp(0) * (torch.min(p_y2,t_y2)-torch.max(p_y1,t_y1)).clamp(0)
        union = w1*h1 + w2*h2 - inter + 1e-7
        iou = inter / union
        return 1 - (iou - rho2/c2 - w_loss - h_loss)

class CIouLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        p_x1,p_y1,p_x2,p_y2 = pred.chunk(4,-1)
        t_x1,t_y1,t_x2,t_y2 = target.chunk(4,-1)
        w1,h1 = p_x2-p_x1, p_y2-p_y1
        w2,h2 = t_x2-t_x1, t_y2-t_y1
        cx_p,cy_p = (p_x1+p_x2)/2, (p_y1+p_y2)/2
        cx_t,cy_t = (t_x1+t_x2)/2, (t_y1+t_y2)/2
        rho2 = (cx_t-cx_p)**2 + (cy_t-cy_p)**2
        cw,ch = torch.max(p_x2,t_x2)-torch.min(p_x1,t_x1), torch.max(p_y2,t_y2)-torch.min(p_y1,t_y1)
        c2 = cw**2 + ch**2 + 1e-7
        v = (4/math.pi**2) * (torch.atan(w2/h2) - torch.atan(w1/h1))**2
        with torch.no_grad():
            alpha = v / (1 - torch.clamp((w1*h1)/(w2*h2+1e-7),0,1) + v + 1e-7)
        inter = (torch.min(p_x2,t_x2)-torch.max(p_x1,t_x1)).clamp(0) * (torch.min(p_y2,t_y2)-torch.max(p_y1,t_y1)).clamp(0)
        union = w1*h1 + w2*h2 - inter + 1e-7
        iou = inter / union
        return 1 - (iou - rho2/c2 - v*alpha)




# ==============================================================
def register_custom_layers():
    import ultralytics.nn.modules as m
    import ultralytics.nn.tasks as t
    from copy import deepcopy

    # ========== 1==========
    custom_modules = {
        "SEAttention": SEAttention,
        "ECAAttention": ECAAttention,
        "CBAM": CBAM,
        "CoordinateAttention": CoordinateAttention,
        "EMA": EMA,
        "SimAM": SimAM,
        "GGCA": GGCA,      
        "FasterNetBlock": FasterNetBlock,  
        "InvertedResidual": InvertedResidual,
        "GhostBottleneck": GhostBottleneck,
        "GhostBottleneckV2": GhostBottleneckV2,  
        "ShuffleBottleneck": ShuffleBottleneck,
        "MBConv": MBConv,
        "C3Ghost": C3Ghost,       
    }
    for name, cls in custom_modules.items():
        setattr(m, name, cls)
        setattr(t, name, cls)

    # ========== 2 ==========
    original_parse_model = t.parse_model

    def make_divisible(x, divisor=8):
        return max(divisor, int(x + divisor / 2) // divisor * divisor)

    def patched_parse_model(cfg, ch=3, verbose=True):
        cfg = deepcopy(cfg)
        scale = cfg.get("scales", {}).get("s", [0.33, 0.50, 1024])
        depth_mult, width_mult, _ = scale
        custom_names = set(custom_modules.keys())

       
        ch_list = [ch]
        all_layers = cfg["backbone"] + cfg["head"]
        for f, n, m, args in all_layers:
            if m == "Concat":
                c_out = sum(ch_list[x] for x in f)
            elif m in custom_names:
                
                c2_base = args[0]
                c_out = make_divisible(c2_base * width_mult)
            else:
                
                c2_base = args[0] if isinstance(args[0], int) else ch_list[-1]
                c_out = make_divisible(c2_base * width_mult) if isinstance(args[0], int) else ch_list[-1]
            ch_list.append(c_out)

        
        idx = 0
        for part in ["backbone", "head"]:
            for i, layer in enumerate(cfg[part]):
                f, n, m, args = layer
                idx += 1
                if m in custom_names:
                    c1 = ch_list[idx - 1]
                    c2 = ch_list[idx]
                    cfg[part][i][3] = [c1, c2] + args[1:]

        return original_parse_model(cfg, ch, verbose)

    t.parse_model = patched_parse_model