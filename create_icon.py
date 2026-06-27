"""Generate elegant app icon for ZnO AI Platform."""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

def draw_icon(size):
    scale = 4  # supersampling for anti-aliasing
    S = size * scale
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy, r = S // 2, S // 2, S // 2 - S // 20

    # ── Background: deep navy circle ────────────────────────────────────────
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(8, 14, 35, 255))

    # ── Gradient overlay rings (simulate radial gradient) ───────────────────
    for i in range(30, 0, -1):
        frac = i / 30
        alpha = int(60 * frac)
        cr = int(r * frac * 0.85)
        blue = int(40 + 60 * frac)
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                     fill=(10, 20, blue, alpha))

    # ── Outer glow ring ─────────────────────────────────────────────────────
    ring_w = max(2, S // 30)
    for offset in range(ring_w + 4, 0, -1):
        a = int(220 * (1 - offset / (ring_w + 4)))
        col = (0, 200 + min(55, offset * 4), 255, a)
        draw.ellipse([cx - r - offset, cy - r - offset,
                      cx + r + offset, cy + r + offset],
                     outline=col, width=1)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(0, 230, 255, 255), width=ring_w)

    # ── Orbital ellipse (tilted) ─────────────────────────────────────────────
    orb_rx, orb_ry = int(r * 0.72), int(r * 0.28)
    orb_img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    orb_draw = ImageDraw.Draw(orb_img)
    for w in range(3, 0, -1):
        a = 80 if w == 1 else 30
        orb_draw.ellipse([cx - orb_rx, cy - orb_ry, cx + orb_rx, cy + orb_ry],
                         outline=(0, 200, 255, a), width=w * scale // 2)
    orb_rotated = orb_img.rotate(30, resample=Image.BICUBIC, center=(cx, cy))
    img = Image.alpha_composite(img, orb_rotated)
    draw = ImageDraw.Draw(img)

    # ── Atom nucleus dot ────────────────────────────────────────────────────
    nd = max(4, S // 22)
    draw.ellipse([cx - nd, cy - nd, cx + nd, cy + nd], fill=(0, 230, 255, 255))
    draw.ellipse([cx - nd // 2, cy - nd // 2, cx + nd // 2, cy + nd // 2],
                 fill=(200, 250, 255, 255))

    # ── "ZnO" text ──────────────────────────────────────────────────────────
    font_size = int(S * 0.30)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    text = "ZnO"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = cx - tw // 2 - bbox[0]
    ty = cy - th // 2 - bbox[1] - S // 25

    # Shadow / glow layers
    for dx, dy, a in [(-3,-3,40),(-2,-2,60),(0,0,100),(2,2,40)]:
        draw.text((tx + dx * scale//4, ty + dy * scale//4),
                  text, fill=(0, 150, 220, a), font=font)

    # Main cyan text
    draw.text((tx, ty), text, fill=(0, 230, 255, 255), font=font)

    # White highlight on top-left of text
    draw.text((tx - scale//4, ty - scale//4),
              text, fill=(180, 245, 255, 60), font=font)

    # ── "AI" subtitle ───────────────────────────────────────────────────────
    sub_size = int(S * 0.10)
    try:
        sub_font = ImageFont.truetype("arialbd.ttf", sub_size)
    except Exception:
        try:
            sub_font = ImageFont.truetype("arial.ttf", sub_size)
        except Exception:
            sub_font = ImageFont.load_default()

    sub = "PLATFORM"
    sbbox = draw.textbbox((0, 0), sub, font=sub_font)
    sw = sbbox[2] - sbbox[0]
    sx = cx - sw // 2
    sy = ty + th + S // 20
    draw.text((sx, sy), sub, fill=(120, 80, 220, 200), font=sub_font)

    # ── Clip to circle ───────────────────────────────────────────────────────
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    img.putalpha(mask)

    # Downscale with high-quality resampling (anti-alias)
    return img.resize((size, size), Image.LANCZOS)


def create_icon():
    sizes = [16, 32, 48, 64, 128, 256]
    images = [draw_icon(s) for s in sizes]
    out = str(Path(__file__).resolve().parent / "zno_platform.ico")
    images[0].save(out, format="ICO",
                   sizes=[(s, s) for s in sizes],
                   append_images=images[1:])
    print(f"Icon saved: {out}")

if __name__ == "__main__":
    create_icon()
