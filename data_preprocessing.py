# data_preprocessing.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
from scipy.signal import stft
from tqdm import tqdm
from config import IMG_SIZE, DATA_DIR, DATASET_DIR, NUM_CLASSES

# 设置中文字体（Windows 示例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def process_bin(bin_path, json_path):
    """处理单个IQ数据文件，生成时频图和YOLO标注"""
    try:
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
        
        # 读取标注
        with open(json_path, 'r') as f_json:
            labels = json.load(f_json)
        
        # 使用实际的观测范围
        f_min, f_max = labels['observation_range']
        # 计算信号总时长（取所有信号的最大结束时间）
        t_max = max(sig['end_time'] for sig in labels['signals']) if labels['signals'] else 0.1
        
        # 生成YOLO格式标注（归一化坐标）
        base_name = os.path.splitext(os.path.basename(bin_path))[0]
        txt_path = os.path.join(os.path.dirname(bin_path), f"{base_name}.txt")
        with open(txt_path, 'w') as f_out:
            for sig in labels['signals']:
                # 计算中心点和宽高（归一化到0-1）
                f_center = (sig['start_frequency'] + sig['end_frequency']) / 2
                t_center = (sig['start_time'] + sig['end_time']) / 2
                f_span = sig['end_frequency'] - sig['start_frequency']
                t_span = sig['end_time'] - sig['start_time']
                
                # 归一化坐标
                x_center = (f_center - f_min) / (f_max - f_min)  # 频率归一化
                y_center = t_center / t_max                      # 时间归一化
                width = f_span / (f_max - f_min)
                height = t_span / t_max
                
                class_id = sig['class']
                # YOLO格式: class x_center y_center width height
                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return heatmap, t, f, t_max, f_min, f_max
    
    except Exception as e:
        print(f"处理文件 {bin_path} 失败: {str(e)}")
        return None, None, None, None, None, None

def build_dataset():
    """构建训练/验证数据集，划分数据并保存图像和标签"""
    # 创建目录结构
    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "test"), exist_ok=True)
    
    # 获取所有原始文件
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.bin')]
    print(f"发现原始文件数量: {len(all_files)}")
    
    # 划分训练集（80%）和验证集（20%）
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    success_count = 0
    fail_list = []
    
    # 处理每个文件
    for filename in tqdm(all_files, desc="处理文件"):
        if not filename.endswith('.bin'):
            continue
        
        bin_path = os.path.join(DATA_DIR, filename)
        json_path = os.path.join(DATA_DIR, filename.replace('.bin', '.json'))
        
        # 检查标注文件是否存在
        if not os.path.exists(json_path):
            print(f"跳过 {filename}: 缺少JSON标注")
            continue
        
        try:
            # 处理信号数据并生成标注
            heatmap, t, f, t_max, f_min, f_max = process_bin(bin_path, json_path)
            if heatmap is None:
                raise RuntimeError("信号处理失败")
            
            # 生成图像和标签文件名
            base_name = os.path.splitext(filename)[0]
            img_name = f"{base_name}.png"
            txt_name = f"{base_name}.txt"
            
            # 确定保存目录（训练/验证）
            subset = "train" if filename in train_files else "val"
            
            # 保存时频图（正方形）
            img_save_path = os.path.join(DATASET_DIR, "images", subset, img_name)
            fig = plt.figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(t, f, 10 * np.log10(heatmap), 
                             shading='gouraud', cmap='viridis', vmin=-120, vmax=0)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(img_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 复制标注文件到对应目录
            src_txt_path = os.path.join(DATA_DIR, f"{base_name}.txt")
            dst_txt_path = os.path.join(DATASET_DIR, "labels", subset, txt_name)
            shutil.copyfile(src_txt_path, dst_txt_path)
            
            success_count += 1
        
        except Exception as e:
            print(f"处理 {filename} 失败: {str(e)}")
            fail_list.append(filename)
    
    # 输出统计结果
    print("\n数据集构建完成:")
    print(f"总文件数: {len(all_files)}")
    print(f"成功处理: {success_count}")
    print(f"失败文件: {len(fail_list)}")
    if fail_list:
        print(f"失败列表: {fail_list}")

if __name__ == "__main__":
    build_dataset()