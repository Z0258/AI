import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 无GUI模式
import matplotlib.pyplot as plt
from scipy.signal import stft
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

# 配置参数
MODEL_PATH = 'D:/AI/runs/detect/iq_detection/weights/best.pt'  # 训练好的模型路径
DATA_DIR = "output"  # 原始数据目录
OUTPUT_DIR = "detection_results"  # 输出目录
IMG_SIZE = 416  # 图像尺寸
CONF_THRESH = 0.25  # 置信度阈值
IOU_THRESH = 0.45  # IoU阈值

def process_iq_file(bin_path):
    """处理IQ文件生成时频图并返回物理坐标参数"""
    # 读取IQ数据
    iq_data = np.fromfile(bin_path, dtype=np.float16)
    if len(iq_data) % 2 != 0:
        raise ValueError("IQ数据长度必须为偶数（实部+虚部）")
    
    # 分离实虚部并生成复数信号
    real, imag = iq_data[0::2], iq_data[1::2]
    signal = (real + 1j * imag).astype(np.complex64)
    
    # 生成时频图（STFT）
    f, t, Zxx = stft(signal, fs=1e6, nperseg=256, return_onesided=False)
    heatmap = np.abs(Zxx).astype(np.float32)
    
    return heatmap, t, f, f[0], f[-1], t[-1]

def detect_signals(model, bin_path, json_path, output_dir):
    """检测信号并生成JSON格式结果"""
    # 处理IQ文件
    heatmap, t, f, f_min, f_max, t_max = process_iq_file(bin_path)
    
    # 生成时频图
    base_name = os.path.splitext(os.path.basename(bin_path))[0]
    img_path = os.path.join(output_dir, f"{base_name}.png")
    
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(t, f, 10 * np.log10(heatmap), 
                     shading='gouraud', cmap='viridis', vmin=-120, vmax=0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(img_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 模型推理
    results = model.predict(
        img_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        save_txt=False,
        save_conf=True
    )
    
    # 解析检测结果
    signals = []
    signal_counter = 0
    
    for result in results:
        for box in result.boxes:
            # 获取归一化坐标
            x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
            conf = box.conf.item()
            class_id = int(box.cls.item())
            
            # 转换为物理坐标
            freq_center = x_center * (f_max - f_min) + f_min
            time_center = y_center * t_max
            freq_span = width * (f_max - f_min)
            time_span = height * t_max
            
            # 计算边界
            start_frequency = freq_center - freq_span / 2
            end_frequency = freq_center + freq_span / 2
            start_time = time_center - time_span / 2
            end_time = time_center + time_span / 2
            
            # 添加到信号列表
            signals.append({
                "signal_id": signal_counter,
                "start_frequency": round(float(start_frequency), 2),
                "end_frequency": round(float(end_frequency), 2),
                "start_time": round(float(start_time), 2),
                "end_time": round(float(end_time), 2),
                "class": class_id
            })
            signal_counter += 1
    
    # 构建完整的JSON结构
    result_json = {
        "signals": signals,
        "observation_range": [round(float(f_min), 2), round(float(f_max), 2)]
    }
    
    # 保存JSON文件
    json_output_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_output_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    return result_json

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH).to(device)
    print(f"模型加载完成，使用设备: {device}")
    
    # 获取所有bin文件
    bin_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.bin')]
    print(f"发现 {len(bin_files)} 个待处理文件")
    
    # 处理每个文件
    for bin_file in tqdm(bin_files, desc="处理文件中"):
        bin_path = os.path.join(DATA_DIR, bin_file)
        json_path = os.path.join(DATA_DIR, bin_file.replace('.bin', '.json'))
        
        # 确保对应的JSON文件存在
        if not os.path.exists(json_path):
            print(f"警告: 跳过 {bin_file}，缺少对应的JSON文件")
            continue
        
        try:
            detect_signals(model, bin_path, json_path, OUTPUT_DIR)
        except Exception as e:
            print(f"处理 {bin_file} 时出错: {str(e)}")
    
    print(f"\n处理完成! 结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()