import torch
import torch.nn as nn
import math
from pathlib import Path

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Auto Padding"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class CBS(nn.Module):
    """
    Standard Convolution: Conv2d + BatchNorm + SiLU
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Standard Bottleneck (Residual Block)
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, c_, k[0], 1)
        self.cv2 = CBS(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (YOLOv8 核心模块)
    Faster implementation of CSP Bottleneck
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, 2 * self.c, 1, 1)
        self.cv2 = CBS((2 + n) * self.c, c2, 1)  # output conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        # 1. Split input into two parts
        y = list(self.cv1(x).chunk(2, 1))
        # 2. Forward pass through bottlenecks
        # y[-1] is the part that goes through blocks
        # y.extend(...) adds the outputs of each block to the list
        y.extend(m(y[-1]) for m in self.m)
        # 3. Concat all parts and pass through final conv
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (CSPDarknet 结尾模块)
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class Detect(nn.Module):
    """
    ✅ 手动重写的 Detect 头
    只负责卷积映射，输出纯粹的特征图，绝不进行解码/NMS/变形。
    输出格式严格为: List[Tensor] 其中每个 Tensor shape 为 [B, (reg_max*4 + nc), H, W]
    """
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 层数 (通常3层 P3-P5)
        self.reg_max = 16  # DFL 积分数 (bins = reg_max + 1)
        self.no = nc + (self.reg_max + 1) * 4  # 每个 Anchor 的输出通道数
        
        self.stride = torch.zeros(self.nl)  # 需要外部赋值
        
        # 定义中间层通道数 (参照 YOLOv8 官方逻辑)
        c2s = [max(16, x // 4, (self.reg_max + 1) * 4) for x in ch]
        c3s = [max(x // 2, min(self.nc, 100)) for x in ch]
        
        # 两个分支：cv2 (Box), cv3 (Cls)
        # 结构: Conv -> Conv -> Conv2d(无激活)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * (self.reg_max + 1), 1))
            for x, c2 in zip(ch, c2s)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x, c3 in zip(ch, c3s)
        )

    def forward(self, x):
        # x: [P3, P4, P5] from neck
        res = []
        for i in range(self.nl):
            # 1. 计算 Box 分支
            bboxes = self.cv2[i](x[i])
            # 2. 计算 Cls 分支
            scores = self.cv3[i](x[i])
            # 3. 直接拼接，不做任何 reshape
            # 输出: [B, 64+nc, H, W]
            res.append(torch.cat((bboxes, scores), 1))
        return res


# ==============================================================================
# 2. 纯手写 YOLOv8 Backbone (CSPDarknet)
# ==============================================================================

class YOLOv8Backbone(nn.Module):
    def __init__(self, width=0.5, depth=0.33, ratio=2.0):
        """
        参数默认为 YOLOv8s (Small) 的配置
        width: 通道缩放系数 (s=0.5)
        depth: 深度(层数)缩放系数 (s=0.33)
        ratio: C2f 模块的通道扩张率
        """
        super().__init__()
        
        # --- 缩放辅助函数 ---
        def make_divisible(x, divisor):
            # 确保通道数是 8 的倍数 (硬件友好)
            return math.ceil(x / divisor) * divisor

        def get_depth(n):
            return max(round(n * depth), 1) if n > 1 else n

        def get_width(c):
            return make_divisible(c * width, 8)

        # --- 网络结构定义 (P1 -> P5) ---
        
        # Layer 0: Stem (P1)
        # 3 -> 64 (s: 32)
        self.stem = CBS(3, get_width(64), 3, 2) 

        # Layer 1: Conv
        # 64 -> 128 (s: 64)
        self.conv1 = CBS(get_width(64), get_width(128), 3, 2)
        
        # Layer 2: C2f (P2)
        # s: n=3 -> n=1
        self.c2f1 = C2f(get_width(128), get_width(128), n=get_depth(3), shortcut=True)

        # Layer 3: Conv
        # 128 -> 256 (s: 128)
        self.conv2 = CBS(get_width(128), get_width(256), 3, 2)

        # Layer 4: C2f (P3 - 输出特征图 1)
        # s: n=6 -> n=2
        self.c2f2 = C2f(get_width(256), get_width(256), n=get_depth(6), shortcut=True)

        # Layer 5: Conv
        # 256 -> 512 (s: 256)
        self.conv3 = CBS(get_width(256), get_width(512), 3, 2)

        # Layer 6: C2f (P4 - 输出特征图 2)
        # s: n=6 -> n=2
        self.c2f3 = C2f(get_width(512), get_width(512), n=get_depth(6), shortcut=True)

        # Layer 7: Conv
        # 512 -> 1024 (s: 512)
        self.conv4 = CBS(get_width(512), get_width(1024), 3, 2)

        # Layer 8: C2f (P5)
        # s: n=3 -> n=1
        self.c2f4 = C2f(get_width(1024), get_width(1024), n=get_depth(3), shortcut=True)

        # Layer 9: SPPF (P5 - 输出特征图 3)
        self.sppf = SPPF(get_width(1024), get_width(1024), k=5)
        
        # 记录输出通道数，方便外部调用
        self.out_channels = [
            get_width(256),  # P3
            get_width(512),  # P4
            get_width(1024)  # P5
        ]
        
        # 初始化权重
        self._init_weights()

    def forward(self, x):
        # P1/P2 (通常不用作检测特征)
        x = self.stem(x)
        x = self.conv1(x)
        x = self.c2f1(x)
        
        # P3
        x = self.conv2(x)
        x = self.c2f2(x)
        feat_p3 = x  # Save P3
        
        # P4
        x = self.conv3(x)
        x = self.c2f3(x)
        feat_p4 = x  # Save P4
        
        # P5
        x = self.conv4(x)
        x = self.c2f4(x)
        x = self.sppf(x)
        feat_p5 = x  # Save P5
        
        # 返回标准的特征金字塔列表
        return [feat_p3, feat_p4, feat_p5]

    def _init_weights(self):
        """Kaiming Initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, weight_path='yolov8s.pt'):
        """
        Load YOLOv8s pretrained backbone weights from Ultralytics checkpoint.
        """
        ckpt_path = Path(weight_path)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {ckpt_path}")

        from ultralytics import YOLO

        yolo_model = YOLO(str(ckpt_path), task='detect').model
        src_layers = yolo_model.model[:10]
        dst_layers = [
            self.stem,
            self.conv1,
            self.c2f1,
            self.conv2,
            self.c2f2,
            self.conv3,
            self.c2f3,
            self.conv4,
            self.c2f4,
            self.sppf,
        ]

        for src, dst in zip(src_layers, dst_layers):
            dst.load_state_dict(src.state_dict(), strict=True)