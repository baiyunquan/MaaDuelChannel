from pathlib import Path

import cv2
import numpy as np

from maa_duel.vision.roster import crop_normalized, default_slot_specs


def extract_clean_empty_slot_rgba(workspace: Path) -> Path:
    prep_frames = list(workspace.glob(".staging/extract/**/*-prep.jpg"))
    if not prep_frames:
        raise FileNotFoundError("No prep frames found in workspace to extract empty slot from")
    
    target_dim = 120
    cx, cy = target_dim // 2, target_dim // 2
    R = int(target_dim * 0.44)  # 52 pixels radius
    
    valid_crops = []
    for pf in prep_frames[:60]:
        img = cv2.imread(str(pf))
        if img is None:
            continue
        specs = default_slot_specs(img.shape[:2])
        crop = crop_normalized(img, specs[2].icon_rect)
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (target_dim, target_dim), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray[35:85, 35:85])) <= 16.0:
            valid_crops.append(resized.astype(np.float32))
            
    print(f"Collected {len(valid_crops)} aligned empty crops for ensemble matting")
    if len(valid_crops) < 5:
        raise RuntimeError("Too few valid empty crops found")
        
    med_crop = np.median(valid_crops, axis=0).astype(np.uint8)
    gray_med = cv2.cvtColor(med_crop, cv2.COLOR_BGR2GRAY)
    
    # Measure authentic disk base color
    disk_bg_level = float(np.median(gray_med[40:80, 40:80]))
    diff = gray_med.astype(np.float32) - disk_bg_level
    
    y, x = np.ogrid[:target_dim, :target_dim]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # Locate authentic slash line
    slash_mask_pts = (dist_from_center < R * 0.70) & (diff > 12.0)
    ys, xs = np.where(slash_mask_pts)
    vx, vy, x0, y0 = cv2.fitLine(np.column_stack((xs, ys)), cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Distance from fitted authentic line
    line_dist = np.abs(vy[0] * (x - x0[0]) - vx[0] * (y - y0[0]))
    
    # Slash along fitted line with authentic length and thickness
    slash_intensity = np.clip(1.0 - line_dist / 2.2, 0.0, 1.0) * (dist_from_center < (R * 0.65)).astype(np.float32)
    
    # Authentic disk alpha: smooth circular boundary
    disk_alpha = np.clip((R - dist_from_center) / 1.5 + 0.5, 0.0, 1.0) * 0.72
    
    # Base dark slate gray color
    base_bgr = np.array([70, 72, 70], dtype=np.float32)
    # Pale white slash color
    slash_bgr = np.array([215, 220, 215], dtype=np.float32)
    
    final_alpha = np.clip(disk_alpha + slash_intensity * (0.95 - disk_alpha), 0.0, 1.0)
    final_bgr = base_bgr * (1.0 - slash_intensity[..., None]) + slash_bgr * slash_intensity[..., None]
    
    # Strict cutoff outside circle
    final_alpha[dist_from_center >= R + 1.0] = 0.0
    
    rgba = np.zeros((target_dim, target_dim, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(final_bgr, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8)
    
    out_path = workspace / "assets" / "ui" / "empty_slot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgba)
    print(f"Generated clean RGBA empty slot asset at: {out_path}")
    return out_path

if __name__ == "__main__":
    import sys
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/data/MAA-DuelChannel/workspace")
    extract_clean_empty_slot_rgba(ws)
