import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from transformers import DebertaV2Tokenizer
from tqdm import tqdm

# 保持类别映射 (虽然 Visual Grounding 通常视为单类，但保留映射有助于扩展)
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

class DIORRSVGDataset(Dataset):
    def __init__(self, data_root, xml_root, split_txt_path, tokenizer_path, img_size=640, max_len=77,
                 min_objects_per_image=1, use_all_images=False, spatial_hint_prob=0.0):
        """
        REC/VG 任务专用 Dataset
        核心逻辑：一个 Sample = 一张图 + 一个特定的 Prompt + 一个特定的 GT Box
        """
        self.data_root = data_root
        self.xml_root = xml_root
        self.img_size = img_size
        self.max_len = max_len
        self.class_to_id = {name: i for i, name in enumerate(DIOR_CLASSES)}
        self.min_objects_per_image = max(int(min_objects_per_image), 1)
        self.use_all_images = bool(use_all_images)
        self.spatial_hint_prob = float(np.clip(spatial_hint_prob, 0.0, 1.0))
        
        # 1. 加载 Tokenizer
        try:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"⚠️ Tokenizer load failed from {tokenizer_path}, trying huggingface hub...")
            self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

        # 2. 读取图片 ID
        if not os.path.exists(split_txt_path):
            raise FileNotFoundError(f"❌ Split file not found: {split_txt_path}")
        
        with open(split_txt_path, 'r') as f:
            img_ids = [x.strip() for x in f.readlines() if x.strip()]

        if self.use_all_images:
            xml_ids = [os.path.splitext(x)[0] for x in os.listdir(self.xml_root) if x.endswith('.xml')]
            xml_ids = sorted(set(xml_ids))
            print(f"⚠️ use_all_images=True: override split file and use all XML ids ({len(xml_ids)} images).")
            img_ids = xml_ids

        # 3. 构建扁平化的 "描述-目标" 索引列表
        self.samples = []
        print(f"Parsing XMLs for Visual Grounding pairs from {os.path.basename(split_txt_path)}...")
        
        # 这一步可能会花点时间，但在 __init__ 里做一次是值得的
        selected_images = 0
        single_obj_images = 0
        multi_obj_images = 0

        for img_id in tqdm(img_ids):
            xml_path = os.path.join(self.xml_root, f"{img_id}.xml")
            if not os.path.exists(xml_path):
                continue
                
            # 解析该图片下的所有 object
            objects = self._parse_xml_for_vg(xml_path)
            
            # 先过滤有效框，再根据每图目标数进行样本筛选
            valid_objects = []
            for obj in objects:
                if obj['box'][2] <= obj['box'][0] or obj['box'][3] <= obj['box'][1]:
                    continue
                valid_objects.append(obj)

            if len(valid_objects) < self.min_objects_per_image:
                continue

            selected_images += 1
            if len(valid_objects) == 1:
                single_obj_images += 1
            else:
                multi_obj_images += 1

            # 将每个有描述的对象拆分为一个独立的样本
            for i, obj in enumerate(valid_objects):
                self.samples.append({
                    'img_id': img_id,
                    'box': obj['box'],      # [xmin, ymin, xmax, ymax]
                    'name': obj['name'],    # 类别名
                    'description': obj['desc'], # 真实文本描述
                    'context': [o for j, o in enumerate(valid_objects) if j != i]
                })
                
        print(f"Loaded {len(self.samples)} referring expressions from {selected_images} selected images.")
        print(f"Image composition | single-target: {single_obj_images}, multi-target: {multi_obj_images}, min_objects_per_image={self.min_objects_per_image} (no forced multi-target bias)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 获取当前样本
        sample = self.samples[index]
        img_id = sample['img_id']
        
        # 1. 加载图片
        img_path = os.path.join(self.data_root, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            # 容错处理：如果读取失败，随机换一个
            print(f"⚠️ Warning: Failed to load {img_path}, using next sample.")
            return self.__getitem__((index + 1) % len(self))

        # 2. Letterbox Resize (保持宽高比)
        img_h, img_w = img.shape[:2]
        img_resized, ratio, pad = self._letterbox(img, new_shape=(self.img_size, self.img_size))
        
        # BGR -> RGB & Normalize -> CHW
        img_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).float() / 255.0

        # 3. 处理 Target Box
        # sample['box'] 是 [xmin, ymin, xmax, ymax] (原图坐标)
        raw_box = np.array(sample['box'])
        
        # 坐标变换：原图 -> Resize后的图
        # box = raw_box * ratio + pad
        box = raw_box.copy()
        box[[0, 2]] *= ratio[0] # x scale
        box[[1, 3]] *= ratio[1] # y scale
        box[[0, 2]] += pad[0]   # x pad
        box[[1, 3]] += pad[1]   # y pad
        
        # 归一化 xywh (YOLO 格式: center_x, center_y, width, height)
        # 注意：这里分母是 resize 后的图大小 (640, 640)
        norm_h, norm_w = img_resized.shape[:2]
        
        x_c = (box[0] + box[2]) / 2 / norm_w
        y_c = (box[1] + box[3]) / 2 / norm_h
        w_b = (box[2] - box[0]) / norm_w
        h_b = (box[3] - box[1]) / norm_h
        if w_b <= 0 or h_b <= 0:
            print(f"⚠️ Warning: Invalid box for {img_id}, using next sample.")
            return self.__getitem__((index + 1) % len(self))
        x_c = float(np.clip(x_c, 0.0, 1.0))
        y_c = float(np.clip(y_c, 0.0, 1.0))
        w_b = float(np.clip(w_b, 1e-6, 1.0))
        h_b = float(np.clip(h_b, 1e-6, 1.0))
        
        cls_name = sample['name']
        cls_id = self.class_to_id.get(cls_name, 0)
        
        # 构造 target: [class_id, x, y, w, h]
        # shape: (1, 5)
        target_boxes = torch.tensor([[cls_id, x_c, y_c, w_b, h_b]], dtype=torch.float32)

        # 4. 处理真实描述 (Description)
        text_prompt = sample['description']
        if not isinstance(text_prompt, str) or text_prompt.strip() == "":
            text_prompt = f"Find the {sample['name']}" # 兜底
            
        if self.spatial_hint_prob > 0.0 and np.random.rand() < self.spatial_hint_prob:
            # 连续空间监督（数值形式）+ 相对关系（与邻近目标）
            context_boxes = []
            for ctx in sample.get('context', []):
                cbox = np.array(ctx['box'], dtype=np.float32).copy()
                cbox[[0, 2]] *= ratio[0]
                cbox[[1, 3]] *= ratio[1]
                cbox[[0, 2]] += pad[0]
                cbox[[1, 3]] += pad[1]

                ocx = float(np.clip((cbox[0] + cbox[2]) / 2 / norm_w, 0.0, 1.0))
                ocy = float(np.clip((cbox[1] + cbox[3]) / 2 / norm_h, 0.0, 1.0))
                ow = float(np.clip((cbox[2] - cbox[0]) / norm_w, 1e-6, 1.0))
                oh = float(np.clip((cbox[3] - cbox[1]) / norm_h, 1e-6, 1.0))
                context_boxes.append([ocx - ow / 2, ocy - oh / 2, ocx + ow / 2, ocy + oh / 2])

            text_prompt = f"{text_prompt} {self._build_spatial_hint(x_c, y_c, w_b, h_b, context_boxes=context_boxes)}"

        encoded = self.tokenizer(
            text_prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attn_mask = encoded['attention_mask'].squeeze(0)
        
        # 5. Spatial GT (归一化中心点 [cx, cy])
        # target_boxes[0, 1:3] 就是 [x_c, y_c]
        spa_gt = target_boxes[0, 1:3].clone() 

        return img_tensor, input_ids, attn_mask, target_boxes, spa_gt


    def _build_spatial_hint(self, x_c, y_c, w_b, h_b, context_boxes=None):
        """
        构造连续几何空间提示（数值约束），而非离散方位词。
        形式示例："[SPATIAL] cx=0.432 cy=0.771 w=0.120 h=0.095 | rel(dx=-0.210,dy=0.084,ow=0.05,oh=0.06)"
        """
        prompt = f"[SPATIAL] cx={x_c:.3f} cy={y_c:.3f} w={w_b:.3f} h={h_b:.3f}"

        if context_boxes:
            rel_tokens = []
            for ob in context_boxes[:3]:  # 最多使用 3 个上下文目标，控制长度
                ox1, oy1, ox2, oy2 = ob
                ocx = (ox1 + ox2) * 0.5
                ocy = (oy1 + oy2) * 0.5
                ow = max(ox2 - ox1, 1e-6)
                oh = max(oy2 - oy1, 1e-6)
                rel_tokens.append(
                    f"rel(dx={x_c - ocx:.3f},dy={y_c - ocy:.3f},ow={ow:.3f},oh={oh:.3f})"
                )
            if rel_tokens:
                prompt = prompt + " | " + " ; ".join(rel_tokens)

        return prompt


    def _parse_xml_for_vg(self, xml_path):
        """解析 XML 获取 bounding box 和 description"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            
            # 提取 description
            desc_node = obj.find("description")
            if desc_node is not None and desc_node.text:
                desc = desc_node.text
            else:
                # 兜底：如果 XML 里没写描述，用类别名造一个
                desc = f"Find the {name}" 
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            objects.append({
                'name': name,
                'box': [xmin, ymin, xmax, ymax],
                'desc': desc
            })
            
        return objects

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        YOLOv5/v8 标准 Letterbox 实现
        Resize image to a 32-pixel-multiple rectangle
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
        
        # 返回: 处理后的图, (宽比例, 高比例), (宽padding, 高padding)
        return img, ratio, (left, top)


def rsvlm_collate_fn(batch):
    """
    Collate function to stack data
    """
    # batch 是一个 list，每个元素是 __getitem__ 返回的 tuple
    # zip(*batch) 会把 list of tuples 转成 tuple of lists
    imgs, input_ids, masks, boxes, spa_gts = zip(*batch)

    # 1. Stack Tensors
    imgs = torch.stack(imgs, 0)          # [B, 3, H, W]
    input_ids = torch.stack(input_ids, 0)# [B, L]
    masks = torch.stack(masks, 0)        # [B, L]
    spa_gts = torch.stack(spa_gts, 0)    # [B, 2]

    # 2. Process Targets for YOLO
    # YOLO 需要 target 格式为: [image_idx, class, x, y, w, h]
    # 这里的 boxes 是一个 tuple，每个元素是 (1, 5) 的 Tensor
    targets = []
    for i, box in enumerate(boxes):
        # 创建索引列 [i]
        batch_idx = torch.full((box.shape[0], 1), i, dtype=box.dtype)
        # 拼接 [idx, cls, x, y, w, h]
        t = torch.cat((batch_idx, box), 1)
        targets.append(t)
        
    targets = torch.cat(targets, 0) # [B, 6]

    return imgs, input_ids, masks, targets, spa_gts