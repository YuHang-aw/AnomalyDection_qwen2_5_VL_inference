import os
from glob import glob

def load_dataset(data_dir: str) -> list[tuple[str, str]]:
    """
    加载数据集，返回 (图片路径, 标签) 的列表。
    标签 'p' 代表正样本 (异常), 'n' 代表负样本 (正常)。
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")

    positive_path = os.path.join(data_dir, 'p')
    negative_path = os.path.join(data_dir, 'n')

    pos_files = glob(os.path.join(positive_path, '*.jpg')) + glob(os.path.join(positive_path, '*.png'))
    neg_files = glob(os.path.join(negative_path, '*.jpg')) + glob(os.path.join(negative_path, '*.png'))

    dataset = [(path, 'p') for path in pos_files] + [(path, 'n') for path in neg_files]
    
    if not dataset:
        print(f"警告: 在目录 {data_dir} 中没有找到任何图片文件。")

    return dataset
