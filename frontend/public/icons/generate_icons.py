"""Generate PWA icons for ZnO Supercapacitor AI Platform."""
import math
from PIL import Image, ImageDraw, ImageFont

BG      = (7, 11, 20)
CYAN    = (0, 212, 255)
CYAN_DIM = (0, 212, 255, 15)

def draw_icon(size: int, safe_pad: int = 0) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img, "RGBA")

    # Background rounded rect
    r = size // 8
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG)

    cx, cy = size // 2, size // 2
    area   = size - 2 * safe_pad

    # Hexagon
    hex_r = area * 0.38
    pts = []
    for i in range(6):
        angle = math.radians(90 + 60 * i)
        pts.append((cx + hex_r * math.cos(angle), cy + hex_r * math.sin(angle)))

    # Filled hex (very dim)
    fill_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill_layer, "RGBA")
    fd.polygon(pts, fill=(0, 212, 255, 18))
    img = Image.alpha_composite(img, fill_layer)
    d   = ImageDraw.Draw(img, "RGBA")

    # Hex outline
    lw = max(2, size // 64)
    for i in range(6):
        d.line([pts[i], pts[(i + 1) % 6]], fill=CYAN, width=lw)

    # CV curve  (anodic hump then cathodic return)
    wave_w  = area * 0.52
    wave_h  = area * 0.22
    x0      = cx - wave_w / 2
    x1      = cx + wave_w / 2
    steps   = 120
    curve_pts = []
    for i in range(steps + 1):
        t = i / steps          # 0 → 1
        x = x0 + t * (x1 - x0)
        # forward sweep: gaussian hump at t≈0.35
        fwd = math.exp(-((t - 0.35) ** 2) / 0.018)
        # reverse sweep: inverted gaussian at t≈0.70
        rev = -math.exp(-((t - 0.70) ** 2) / 0.015)
        y   = cy - wave_h * (fwd + rev)
        curve_pts.append((x, y))

    # Glow pass (wide, dim)
    glow_lw = max(6, size // 28)
    glow_col = (0, 212, 255, 45)
    gl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl, "RGBA")
    gd.line(curve_pts, fill=glow_col, width=glow_lw, joint="curve")
    img = Image.alpha_composite(img, gl)
    d   = ImageDraw.Draw(img, "RGBA")

    # Main curve
    curve_lw = max(3, size // 40)
    d.line(curve_pts, fill=CYAN, width=curve_lw, joint="curve")

    # "ZnO" label
    label_size = max(12, size // 16)
    try:
        font = ImageFont.truetype("arial.ttf", label_size)
    except Exception:
        font = ImageFont.load_default()
    text = "ZnO AI"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ty = cy + hex_r * 0.55
    d.text((cx - tw / 2, ty - th / 2), text, font=font, fill=CYAN)

    return img


# 192 × 192
img192 = draw_icon(192)
img192.save("icon-192.png")
print("icon-192.png  saved")

# 512 × 512
img512 = draw_icon(512)
img512.save("icon-512.png")
print("icon-512.png  saved")

# Maskable 512 × 512  (safe-zone padding = 20% → 51 px each side)
imgM = draw_icon(512, safe_pad=51)
imgM.save("icon-maskable.png")
print("icon-maskable.png saved")
