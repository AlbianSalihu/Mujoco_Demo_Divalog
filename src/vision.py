# src/vision.py
import mujoco
import numpy as np
import cv2

def create_renderer(model, H, W):
    """Renderer offscreen (H, W)."""
    return mujoco.Renderer(model, H, W)

def get_rgb_depth(renderer, data, cam_id: int):
    """
    MuJoCo 3.3.7 : on peut passer l'ID entier à update_scene.
    Retourne (rgb uint8 HxWx3, depth float32 HxW | None).
    """
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()  # (H,W,3) uint8

    depth = None
    if hasattr(renderer, "read_depth"):
        try:
            depth = renderer.read_depth()
        except Exception:
            depth = None
    if depth is None and hasattr(renderer, "read_pixels"):
        try:
            out = renderer.read_pixels(depth=True)
            if isinstance(out, tuple) and len(out) == 2:
                rgb2, depth = out
                if isinstance(rgb2, np.ndarray) and rgb2.shape == rgb.shape:
                    rgb = rgb2
            else:
                depth = out
        except Exception:
            depth = None
    return rgb, depth

def detect_screw_center_bgr(bgr):
    """Détection simple par teintes sombres (modifie si tu colores la vis)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 40), (179, 60, 120))
    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] < 1e-5:
        return None, mask
    u = int(M["m10"]/M["m00"])
    v = int(M["m01"]/M["m00"])
    return (u, v), mask
