# config.py
IMG_SIZE = 416          # 输入图像尺寸
BATCH_SIZE = 8          # 批次大小
EPOCHS = 50            # 训练轮次
LR = 1e-4               # 初始学习率
DATA_DIR = "output"     # 原始数据目录
DATASET_DIR = "iq_dataset"  # 处理后的数据集目录
NUM_CLASSES = 14        # 类别数量 (0-13)