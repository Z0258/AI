# test.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from ultralytics import YOLO
from config import IMG_SIZE, DATA_DIR, DATASET_DIR

def detect_heatmap(model, test_bin):
    """使用训练好的模型检测热力图并可视化结果"""
    try:
        # 1. 生成测试数据的时频图
        iq_data = np.fromfile(test_bin, dtype=np.float16)
        if len(iq_data) % 2 != 0:
            raise ValueError("测试IQ数据长度必须为偶数")
        
        real, imag = iq_data[0::2], iq_data[1::2]
        signal = (real + 1j * imag).astype(np.complex64)
        
        f, t, Zxx = stft(signal, fs=1e6, nperseg=256, return_onesided=False)
        heatmap = np.abs(Zxx).astype(np.float32)
        
        # 读取JSON获取实际范围
        json_path = test_bin.replace('.bin', '.json')
        with open(json_path, 'r') as f_json:
            labels = json.load(f_json)
        f_min, f_max = labels['observation_range']
        t_max = max(sig['end_time'] for sig in labels['signals']) if labels['signals'] else 1.0
        
        # 2. 模型推理
        img_name = os.path.basename(test_bin).replace('.bin', '.png')
        img_path = os.path.join(DATASET_DIR, "images", "test", img_name)
        # 如果测试图像不存在，则生成
        if not os.path.exists(img_path):
            # 创建测试图像目录
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            # 生成并保存
            fig = plt.figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(t, f, 10 * np.log10(heatmap), 
                             shading='gouraud', cmap='viridis', vmin=-120, vmax=0)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(img_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # 进行预测
        results = model.predict(
            img_path,
            save=True,
            imgsz=IMG_SIZE,
            conf=0.25  # 置信度阈值
        )
        
        # 3. 可视化时频图和检测结果
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(heatmap), 
                     shading='gouraud', cmap='magma', vmin=-120, vmax=0)
        plt.colorbar(label='强度 [dB]')
        plt.xlabel('时间 (s)')
        plt.ylabel('频率 (Hz)')
        plt.title('信号检测结果')
        
        # 绘制检测框
        for result in results:
            for box in result.boxes:
                # 获取归一化坐标
                x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                
                # 转换为物理坐标
                time_center = x_center * t_max
                freq_center = y_center * (f_max - f_min) + f_min
                time_span = width * t_max
                freq_span = height * (f_max - f_min)
                
                # 计算框的四个角
                x1 = time_center - time_span/2
                x2 = time_center + time_span/2
                y1 = freq_center - freq_span/2
                y2 = freq_center + freq_span/2
                
                # 绘制矩形框
                plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=1.5)
                # 显示类别和置信度
                class_id = int(box.cls)
                conf = box.conf.item()
                plt.text(x1, y1, f"{class_id}:{conf:.2f}", 
                        color='white', fontsize=10, 
                        bbox=dict(facecolor='red', alpha=0.7))
        
        plt.show()
    
    except Exception as e:
        print(f"检测过程失败: {str(e)}")