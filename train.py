# train.py
import os
import yaml
import torch
from ultralytics import YOLO
from config import IMG_SIZE, BATCH_SIZE, EPOCHS, LR, DATASET_DIR, NUM_CLASSES

def train_model():
    # 创建数据集配置文件
    data_config = {
        'train': os.path.join('D:/AI/iq_dataset/images/train'),
        'val': os.path.join('D:/AI/iq_dataset/images/val'),
        'names': [str(i) for i in range(NUM_CLASSES)],
        'nc': NUM_CLASSES
    }
    
    # 保存YAML配置文件
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    # 初始化模型
    model = YOLO('D:/AI/model/yolov8n.pt') 
    
    # 训练参数
    train_args = {
        'data': yaml_path,
        'imgsz': IMG_SIZE,
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'lr0': LR,
        'name': 'iq_detection',
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'augment': True,  # 启用基础数据增强
        'hsv_h': 0.015,   # 色调增强
        'hsv_s': 0.7,     # 饱和度增强
        'hsv_v': 0.4,     # 亮度增强
        'translate': 0.1, # 平移增强
        'scale': 0.5,     # 缩放增强
    }
    
    # 开始训练
    results = model.train(**train_args)
    print("训练完成! 最佳模型保存在:", results.save_dir)

if __name__ == "__main__":
    train_model()