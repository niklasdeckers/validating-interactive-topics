import os
from flask import Flask, send_file, abort
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np

IMAGE_DIR = "./output/images"
FONT_PATH = "./fonts/DejaVuSans.ttf"  # bundled font
PORT = 8080
MAX_HEIGHT = 1000
BASE_FONT_RATIO = 0.08
BASE_BORDER_RATIO = 0.02
BASE_SPACING_RATIO = 0.03
MIN_BORDER_WIDTH = 2
MIN_SPACING = 5

app = Flask(__name__)



def solve_new_widths(heights, widths, total_width):
    heights = np.asarray(heights, dtype=float)
    widths = np.asarray(widths, dtype=float)

    Hmax = heights.max()
    u = Hmax * widths / heights

    # Sort upper bounds
    u_sorted = np.sort(u)
    cumsum = np.cumsum(u_sorted)

    n = len(u)

    for k in range(n):
        remaining = n - k
        if k == 0:
            lambda_candidate = total_width / remaining
        else:
            lambda_candidate = (total_width - cumsum[k-1]) / remaining

        if lambda_candidate <= u_sorted[k]:
            break
    else:
        # All caps active
        return u.copy().astype(int)

    lambda_star = lambda_candidate
    x = np.minimum(u, lambda_star)
    return x.astype(int)
    

def load_font(size):
    """Try bundled font, then system fonts, fallback to default."""
    if os.path.exists(FONT_PATH):
        try:
            font = ImageFont.truetype(FONT_PATH, size)
            print(f"✅ Using bundled font {FONT_PATH}")
            return font
        except Exception as e:
            print(f"⚠️ Failed to load bundled font: {e}")

    # Try system fonts
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
        print("✅ Using system DejaVuSans.ttf")
        return font
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size)
            print("✅ Using system arial.ttf")
            return font
        except Exception:
            print("⚠️ No TTF font found, using default tiny font")
            return ImageFont.load_default()

def stitch_images(image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    n = len(images)

    if n == 1:
        # Single image, no border or label
        img = images[0]
        stitched = Image.new("RGB", (img.width, img.height), color="white")
        stitched.paste(img, (0, 0))
        return stitched

    # Step 1: initial scaling to cap height at MAX_HEIGHT
    base_height = min(MAX_HEIGHT, max(img.height for img in images))
    scaled_images = []
    for img in images:
        scale = base_height / img.height
        w = max(1, int(img.width * scale))
        h = max(1, int(img.height * scale))
        scaled_images.append(img.resize((w, h)))

    # Step 2: preliminary final row height
    prelim_heights = [img.height for img in scaled_images]
    prelim_widths = [img.width for img in scaled_images]
    final_row_height = max(prelim_heights)

    # Step 3: adaptive sizes based on final row height
    font_size = max(10, int(final_row_height * BASE_FONT_RATIO))
    font = load_font(font_size)
    border_width = max(MIN_BORDER_WIDTH, int(final_row_height * BASE_BORDER_RATIO))
    spacing = max(MIN_SPACING, int(final_row_height * BASE_SPACING_RATIO))
    corner_radius = border_width
    label_height = font_size + 10 if n <= 9 else 0

    # Step 4: target row width for n:1
    target_width = n * final_row_height

    # Step 5: check if total width exceeds target, apply aspect-ratio-weighted scaling
    total_width = sum(prelim_widths) + 2*border_width*n + spacing*(n-1)
    if total_width > target_width:
        scaled_widths = solve_new_widths([img.height for img in scaled_images], [img.width for img in scaled_images], target_width)

        # Resize images proportionally to preserve aspect ratios
        for i, img in enumerate(scaled_images):
            orig_w, orig_h = img.size
            new_w = scaled_widths[i]
            new_h = max(1, int(orig_h * new_w / orig_w))  # preserve aspect ratio
            scaled_images[i] = img.resize((new_w, new_h))

    # Step 6: recompute final canvas size
    widths = [img.width for img in scaled_images]
    total_width = sum(widths) + 2*border_width*n + spacing*(n-1)
    final_height = final_row_height + label_height + 2*border_width

    # Step 7: border colors
    border_colors = ["#0062B1", "#0C797D"] if n == 2 else ["#73D8FF"] * n

    # Step 8: create canvas
    stitched = Image.new("RGB", (total_width, final_height), color="white")
    draw = ImageDraw.Draw(stitched)

    # Step 9: paste images with borders and number labels
    x_offset = 0
    for idx, img in enumerate(scaled_images):
        bw = border_width
        img_w, img_h = img.width, img.height
        y_offset = (final_height - label_height - 2*bw - img_h) // 2

        # Draw rounded border
        draw.rounded_rectangle(
            [x_offset, y_offset, x_offset + img_w + 2*bw - 1, y_offset + img_h + 2*bw - 1],
            radius=corner_radius,
            fill=border_colors[idx]
        )

        # Paste image
        stitched.paste(img, (x_offset + bw, y_offset + bw))

        # Draw number label
        if n <= 9:
            number = str(idx + 1)
            bbox = draw.textbbox((0,0), number, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            text_x = x_offset + bw + (img_w - w)//2
            text_y = y_offset + img_h + 2*bw + 2
            draw.text((text_x, text_y), number, fill="black", font=font)

        x_offset += img_w + 2*bw + spacing

    return stitched



@app.route("/<filename>")
def handle_request(filename):
    if not filename.endswith(".jpg"):
        abort(400, "Only .jpg requests are supported")

    stem = filename[:-4]
    ids = stem.split("_")

    if len(ids) < 1:
        abort(400, "At least one image id is required")

    image_paths = [os.path.join(IMAGE_DIR, f"{img_id}_image.jpg") for img_id in ids]
    for path in image_paths:
        if not os.path.exists(path):
            abort(404, f"Missing source image: {os.path.basename(path)}")

    stitched = stitch_images(image_paths)

    buf = BytesIO()
    stitched.save(buf, format="JPEG")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/jpeg",
        download_name=filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
