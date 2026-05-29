#!/usr/bin/env python3
"""
6孔OMR刺激视频生成器
配色：红色条纹(255,0,0) + 洋红色背景(160,0,255)
"""
import os
import numpy as np
from moviepy import VideoClip

def cm_to_px(cm, dpi):
    return int(round((cm / 2.54) * dpi))

def make_circle_centers_no_spacing(w_px, h_px, rows=2, cols=3):
    """
    当直径刚好填满空间时（4cm * 3 = 12cm, 4cm * 2 = 8cm）
    圆心位置就在 1/6, 3/6, 5/6 (宽度) 和 1/4, 3/4 (高度) 处。
    """
    centers = []
    for r in range(rows):
        cy = int(round(h_px * (2*r + 1) / (2 * rows)))
        for c in range(cols):
            cx = int(round(w_px * (2 * c + 1) / (2 * cols)))
            centers.append((cx, cy))
    return centers

def generate_translate_frame(t, w, h, centers, hole_r_px, stripe_period_px, temporal_freq,
                             direction=1, bg_color=(255, 0, 160), stim_color=(0, 0, 255)):
    """
    生成平移条纹帧
    bg_color: 背景颜色 (BGR格式)
    stim_color: 条纹颜色 (BGR格式)
    """
    frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
    speed_px_per_s = stripe_period_px * temporal_freq
    offset = (direction * speed_px_per_s * t) % (2 * stripe_period_px)

    # 预先生成条纹纹理块
    tile_h = (hole_r_px * 2) + 2 * stripe_period_px
    ys = np.arange(tile_h).reshape((-1, 1))
    stripe_mask = ((np.floor((ys + offset) / stripe_period_px) % 2) == 1)
    
    for cx, cy in centers:
        x0, x1 = max(0, cx - hole_r_px), min(w, cx + hole_r_px)
        y0, y1 = max(0, cy - hole_r_px), min(h, cy + hole_r_px)
        hh, ww = y1 - y0, x1 - x0
        
        # 提取条纹遮罩并扩展到3通道
        local_stripe_mask = np.repeat(
            np.repeat(stripe_mask[stripe_period_px:stripe_period_px+hh], ww, axis=1)[:,:,np.newaxis], 
            3, axis=2
        )
        
        # 制作圆形遮罩
        yy, xx = np.ogrid[0:hh, 0:ww]
        circle_mask = ((yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2) <= (hole_r_px ** 2)
        
        # 应用颜色：条纹处用stim_color，否则用bg_color
        patch = np.where(local_stripe_mask, stim_color, bg_color)
        
        # 只在圆形遮罩范围内绘制
        frame[y0:y1, x0:x1][circle_mask] = patch[circle_mask]
    
    return frame

def generate_radial_frame(t, w, h, centers, hole_r_px, angular_period_deg, rotations_in_30s,
                          direction=1, bg_color=(255, 0, 160), stim_color=(0, 0, 255)):
    """
    生成径向条纹帧
    bg_color: 背景颜色 (BGR格式)
    stim_color: 条纹颜色 (BGR格式)
    """
    frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
    ang_speed = direction * (rotations_in_30s * 360.0 / 30.0)
    angle_offset = ang_speed * t

    for cx, cy in centers:
        x0, x1 = max(0, cx - hole_r_px), min(w, cx + hole_r_px)
        y0, y1 = max(0, cy - hole_r_px), min(h, cy + hole_r_px)
        hh, ww = y1 - y0, x1 - x0
        
        yy, xx = np.indices((hh, ww))
        sub_x = (xx + x0) - cx
        sub_y = (yy + y0) - cy
        
        # 计算角度并应用旋转
        theta = (np.degrees(np.arctan2(sub_y, sub_x)) + 360.0 + angle_offset) % 360.0
        stripe_mask = ((np.floor(theta / angular_period_deg) % 2) == 1)
        circle_mask = (sub_x ** 2 + sub_y ** 2) <= (hole_r_px ** 2)
        
        # 应用颜色
        for i in range(3):
            channel = np.where(
                circle_mask,
                np.where(stripe_mask, stim_color[i], bg_color[i]),
                frame[y0:y1, x0:x1, i]
            )
            frame[y0:y1, x0:x1, i] = channel
    
    return frame

def build_video(outpath, mode, width_cm, height_cm, dpi, hole_diameter_cm,
                stripe_mm, temporal_freq, rotations_in_30s, fps=30,
                bg_color=(255, 0, 160), stim_color=(0, 0, 255)):
    """
    构建视频
    bg_color: 背景颜色 (BGR格式)
    stim_color: 条纹颜色 (BGR格式)
    """
    w_px = cm_to_px(width_cm, dpi)
    h_px = cm_to_px(height_cm, dpi)
    hole_r_px = cm_to_px(hole_diameter_cm / 2.0, dpi)
    stripe_period_px = max(2, int(round((stripe_mm / 10.0) * (dpi / 2.54))))
    
    # 获取圆心（无缝排布）
    centers = make_circle_centers_no_spacing(w_px, h_px, rows=2, cols=3)

    def make_frame(t):
        # 30-60秒：静态背景色
        if 30.0 <= t < 60.0:
            return np.full((h_px, w_px, 3), bg_color, dtype=np.uint8)
        
        # 0-30秒：正向运动，60-90秒：反向运动
        direction = 1 if t < 30.0 else -1
        curr_t = t if t < 30.0 else (t - 60.0)

        if mode == 'translate':
            return generate_translate_frame(
                curr_t, w_px, h_px, centers, hole_r_px,
                stripe_period_px, temporal_freq, direction,
                bg_color=bg_color, stim_color=stim_color
            )
        else:
            return generate_radial_frame(
                curr_t, w_px, h_px, centers, hole_r_px,
                angular_period_deg=20, rotations_in_30s=rotations_in_30s,
                direction=direction, bg_color=bg_color, stim_color=stim_color
            )

    clip = VideoClip(lambda tt: make_frame(tt), duration=90.0)
    clip.write_videofile(outpath, codec='libx264', audio=False, fps=fps, preset='ultrafast')

def main():
    # 硬件参数
    WIDTH = 12.0
    HEIGHT = 8.0
    HOLE_DIAMETER = 4.0
    DPI = 264.0  # iPad Air/Pro 标准
    
    # 颜色定义 (BGR格式 - OpenCV/MoviePy使用BGR而非RGB)
    # 红色条纹: RGB(255, 0, 0) -> BGR(0, 0, 255)
    # 洋红色背景: RGB(160, 0, 255) -> BGR(255, 0, 160)
    BG_COLOR = (255, 0, 160)    # 洋红色背景
    STIM_COLOR = (0, 0, 255)    # 红色条纹
    
    print("=" * 70)
    print("6孔OMR刺激视频生成器 - 红色条纹+洋红色背景")
    print("=" * 70)
    print(f"布局: {WIDTH}×{HEIGHT}cm, 孔径: {HOLE_DIAMETER}cm (无间距)")
    print(f"背景色: RGB(160, 0, 255) 洋红色")
    print(f"条纹色: RGB(255, 0, 0) 红色")
    print("=" * 70)

    if not os.path.exists("output"):
        os.makedirs("output")

    # 刺激参数
    stripe_mm = 8.0
    temporal_freq = 0.5  # Hz (慢速)
    rotations_in_30s = 0.5  # 30秒内半圈（慢速）
    speed_cm_s = temporal_freq * (stripe_mm / 10.0)
    
    print(f"条纹宽度: {stripe_mm} mm")
    print(f"时间频率: {temporal_freq} Hz")
    print(f"线性速度: ~{speed_cm_s:.2f} cm/s")
    print(f"旋转速度: {rotations_in_30s} 圈/30秒")
    print("=" * 70)

    # 生成平移条纹视频
    print("\n生成平移条纹视频...")
    build_video(
        "output/omr_translate_red_magenta.mp4", 
        'translate', 
        WIDTH, HEIGHT, DPI, HOLE_DIAMETER, 
        stripe_mm, temporal_freq, rotations_in_30s,
        bg_color=BG_COLOR,
        stim_color=STIM_COLOR
    )
    
    # 生成径向条纹视频
    print("\n生成径向条纹视频...")
    build_video(
        "output/omr_radial_red_magenta.mp4", 
        'radial', 
        WIDTH, HEIGHT, DPI, HOLE_DIAMETER, 
        stripe_mm, temporal_freq, rotations_in_30s,
        bg_color=BG_COLOR,
        stim_color=STIM_COLOR
    )
    
    print("\n" + "=" * 70)
    print("✓ 生成完成！请查看 output 文件夹")
    print("=" * 70)
    print("\n输出文件:")
    print("  - output/omr_translate_red_magenta.mp4 (平移条纹)")
    print("  - output/omr_radial_red_magenta.mp4 (径向条纹)")
    print("\n视频结构:")
    print("  0-30s:   正向运动")
    print("  30-60s:  静止（纯背景色）")
    print("  60-90s:  反向运动")
    print("=" * 70)

if __name__ == '__main__':
    main()