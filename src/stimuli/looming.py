#!/usr/bin/env python3
"""
捕食者拦截 (Looming) 刺激视频生成器
荧光绿背景 + 黑色蜻蜓剪影 (居中)
每5秒从中心指数级放大
"""
import cv2
import numpy as np
import math
import os

class LoomingStimulus:
    def __init__(self, width=1920, height=1080, fps=60):
        self.width = width
        self.height = height
        self.fps = fps
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.WHITE = (255, 255, 255)
    
    def draw_dragonfly(self, frame, cx, cy, size):
        """绘制蜻蜓剪影"""
        # 身体
        body_w, body_h = int(size * 0.2), int(size * 0.6)
        cv2.ellipse(frame, (cx, cy), (body_w, body_h), 0, 0, 360, self.BLACK, -1)
        
        # 头部
        head_r = int(size * 0.15)
        cv2.circle(frame, (cx, cy - body_h - head_r // 2), head_r, self.BLACK, -1)
        
        # 翅膀
        for side in [-1, 1]:  # 左右
            # 上翅膀
            wx = cx + side * int(size * 0.4)
            cv2.ellipse(frame, (wx, cy - int(size * 0.2)), 
                       (int(size * 0.4), int(size * 0.3)),
                       -30 * side, 0, 360, self.BLACK, -1)
            # 下翅膀
            wx = cx + side * int(size * 0.35)
            cv2.ellipse(frame, (wx, cy + int(size * 0.1)), 
                       (int(size * 0.35), int(size * 0.25)),
                       -25 * side, 0, 360, self.BLACK, -1)
        
        # 尾部节段
        for i in range(5):
            seg_y = cy + body_h + i * (size * 0.1)
            seg_w = int(body_w * (1 - i * 0.15))
            if seg_w > 2:
                cv2.circle(frame, (cx, int(seg_y)), seg_w, self.BLACK, -1)
    
    def create_video(self, output_path='output/looming_centered.mp4', duration=60.0):
        """生成looming刺激视频"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        total_frames = int(duration * self.fps)
        looming_interval = 5 * self.fps      # 每5秒
        looming_duration = 0.25 * self.fps   # 250ms扩张
        
        print(f"\n生成视频: {output_path}")
        print(f"分辨率: {self.width}×{self.height} | 时长: {duration}s | 触发: 每5秒")
        
        center_x, center_y = self.width // 2, self.height // 2
        
        for frame_num in range(total_frames):
            # 荧光绿背景
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :, 1] = 255
            
            # Looming效果
            local_frame = frame_num % looming_interval
            if local_frame < looming_duration:
                progress = local_frame / looming_duration
                scale = (math.exp(progress * 5) - 1) / (math.exp(5) - 1)
                size = int(50 + scale * 600)  # 50px → 650px
                self.draw_dragonfly(frame, center_x, center_y, size)
            
            out.write(frame)
            
            if frame_num % (self.fps * 5) == 0:
                print(f"进度: {frame_num / total_frames * 100:.1f}%")
        
        out.release()
        print(f"✓ 完成: {output_path}\n")

if __name__ == "__main__":
    generator = LoomingStimulus(width=1920, height=1080, fps=60)
    generator.create_video('output/looming_centered.mp4', duration=60.0)