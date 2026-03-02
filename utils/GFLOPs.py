import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

# 路径设置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append("/home/bxc/RSLIF")

import torch
from fvcore.nn import FlopCountAnalysis
from models.fusion_module_GroundingQueryInjector import RemoteSensingVLM

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def profile_step(name, module, *inputs):
    """通用的单步分段 GFLOPs 监测函数"""
    try:
        with torch.no_grad():
            # fvcore 需要 inputs 是 tuple 格式
            flops = FlopCountAnalysis(module, inputs)
            # 也可以针对某些层关闭不支持的算子警告
            flops = flops.total() / 1e9  # 转换为 GFLOPs
        print(f"[{name:<25}] GFLOPs: {flops:.3f}")
        return flops
    except Exception as e:
        print(f"[{name:<25}] Failed: {e}")
        return 0.0

def main():
    torch.cuda.empty_cache()
    
    # 1. 初始化模型（确保 cfg 路径正确，或使用默认配置）
    cfg = {'deberta_path': '/home/bxc/RSLIF/pretrain_ckp/deberta-v3-small', 'nc': 20}
    model = RemoteSensingVLM(cfg).to(device)
    model.eval()

    # 2. 准备模拟输入
    batch_size = 1
    imgs = torch.randn(batch_size, 3, 640, 640).to(device)
    # 模拟文本输入 (input_ids, mask)
    input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
    masks = torch.ones((batch_size, 32)).to(device)
    texts = [input_ids, masks]

    print(f"{'='*50}\nLayer-wise GFLOPs Analysis\n{'='*50}")

    with torch.no_grad():
        # --- 分段 1: Text Encoder ---
        # 提取文本特征，这部分通常计算量较小但不可忽略
        flops_text = profile_step("Text_Encoder", model.text_encoder, texts)
        txt_out = model.text_encoder(texts)
        # 模拟 combined_text 为 sem + spa + attr
        text_seq = txt_out['sem'] + txt_out['spa'] + txt_out['attr'] 
        text_vec = text_seq.mean(dim=1)
        mask = txt_out['mask']

        # --- 分段 2: Backbone ---
        flops_backbone = profile_step("Backbone_YOLO", model.backbone, imgs)
        raw_feats = model.backbone(imgs) # raw_p3, raw_p4, raw_p5

        # --- 分段 3: Fusion Neck (关键部分) ---
        # 我们最新的 Neck 接收 (p3, p4, p5, text_seq, text_vec, mask)
        # 注意：fvcore 监测时，inputs 必须与 forward 参数顺序一致
        flops_neck = profile_step("Hybrid_Fusion_Neck", model.neck, 
                                  raw_feats[0], raw_feats[1], raw_feats[2], 
                                  text_seq, mask)
        fused_feats = model.neck(raw_feats[0], raw_feats[1], raw_feats[2], text_seq, mask)

        # --- 分段 4: Detection Head ---
        flops_head = profile_step("Detection_Head", model.head, fused_feats)
        
        # --- 分段 5: Alignment Projector (可选) ---
        flops_proj = profile_step("Alignment_Projector", model.vis_projector, fused_feats[-1])

    # 3. 汇总计算
    total = sum([flops_text, flops_backbone, flops_neck, flops_head, flops_proj])
    
    print(f"{'='*50}")
    print(f"Summary:")
    print(f" - Text Branch:     {flops_text:.3f} G")
    print(f" - Vision Backbone: {flops_backbone:.3f} G")
    print(f" - Multi-modal Neck:{flops_neck:.3f} G")
    print(f" - Detection Head:  {flops_head:.3f} G")
    print(f"{'='*50}")
    print(f"[Total System GFLOPs]: {total:.2f} G")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()