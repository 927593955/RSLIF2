import os
import random

def split_dataset(folder_path, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # 1. 获取目录下所有 .jpg 结尾的文件名（不含后缀）
    files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    if not files:
        print("未在指定文件夹中找到 .jpg 文件。")
        return

    # 2. 打乱顺序
    random.seed(42)  # 固定随机种子，确保结果可复现
    random.shuffle(files)

    # 3. 计算切分索引
    total_count = len(files)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    # 4. 划分数据集
    train_list = files[:train_end]
    val_list = files[train_end:val_end]
    test_list = files[val_end:]

    # 5. 写入 txt 文档
    datasets = {
        "train.txt": train_list,
        "val.txt": val_list,
        "test.txt": test_list
    }

    for filename, data in datasets.items():
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item + '\n')
        print(f"成功写入 {filename}，共 {len(data)} 条数据。")

# --- 使用设置 ---
folder_path = '/data/bxc/DIOR-RSVG/JPEGImages'  # 替换成你的文件夹路径
split_dataset(folder_path)