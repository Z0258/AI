# main.py
from data_preprocessing import build_dataset
from train import train_model
from test import detect_heatmap
import os
from config import DATASET_DIR, DATA_DIR

def main():
    # 步骤1：构建数据集
    #print("===== 开始构建数据集 =====")
    #build_dataset()
    
    # 步骤2：训练模型
    #print("\n===== 开始模型训练 =====")
    #train_model()
    
    # 步骤3：执行测试检测
    print("\n===== 开始目标检测测试 =====")
    # 加载训练好的模型
    model_path = os.path.join('D:/AI/runs/detect/iq_detection/weights/best.pt')
    if not os.path.exists(model_path):
        print(f"错误：模型未找到，路径: {model_path}")
        return
    
    # 测试文件（使用验证集中的一个文件）
    test_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.bin')]
    if not test_files:
        print("错误：没有测试文件")
        return
    test_bin = os.path.join(DATA_DIR, test_files[0])
    
    # 执行检测
    from test import detect_heatmap
    from ultralytics import YOLO
    model = YOLO(model_path)
    detect_heatmap(model, test_bin)

if __name__ == "__main__":
    main()

    