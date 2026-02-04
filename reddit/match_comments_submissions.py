import os
import json
import requests
import pandas as pd
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
SUBMISSIONS_PATH = "submissions.jsonl"
COMMENTS_PATH = "comments.jsonl"

OUT_DIR = "output"
IMG_DIR = os.path.join(OUT_DIR, "images")
META_DIR = os.path.join(OUT_DIR, "meta")
COMMENTS_DIR = os.path.join(OUT_DIR, "comments")

FAILED_IMAGES_PATH = os.path.join(OUT_DIR, "failed_images.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(COMMENTS_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "reddit-dataset-processing/1.0"
}

# -------------------------
# Helpers
# -------------------------
def normalize_imgur_url(url: str) -> str:
    """
    Convert imgur page URLs to direct image URLs if possible.
    """
    if "imgur.com" not in url:
        return url

    if any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]):
        return url

    img_id = url.rstrip("/").split("/")[-1]
    return f"https://i.imgur.com/{img_id}.jpg"


def normalize_parent_id(pid: str) -> str:
    """
    Remove t3_ prefix if present.
    """
    return pid.replace("t3_", "")


def log_failed_image(entry: dict):
    with open(FAILED_IMAGES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def download_image(url: str, out_path: str, submission_id: str) -> bool:
    """
    Download image and convert to RGB JPEG.
    Logs detailed failures for later inspection.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
    except requests.RequestException as e:
        log_failed_image({
            "submission_id": submission_id,
            "url": url,
            "stage": "request",
            "error": type(e).__name__,
            "message": str(e),
        })
        return False

    if resp.status_code != 200:
        log_failed_image({
            "submission_id": submission_id,
            "url": url,
            "stage": "http",
            "status_code": resp.status_code,
        })
        return False

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        log_failed_image({
            "submission_id": submission_id,
            "url": url,
            "stage": "content-type",
            "content_type": content_type,
        })
        return False

    try:
        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")
        img.save(out_path, format="JPEG")
        return True

    except UnidentifiedImageError as e:
        log_failed_image({
            "submission_id": submission_id,
            "url": url,
            "stage": "parse",
            "error": "UnidentifiedImageError",
            "message": str(e),
        })
        return False

    except Exception as e:
        log_failed_image({
            "submission_id": submission_id,
            "url": url,
            "stage": "parse",
            "error": type(e).__name__,
            "message": str(e),
        })
        return False

# -------------------------
# Load data
# -------------------------
submissions = pd.read_json(SUBMISSIONS_PATH, lines=True)
comments = pd.read_json(COMMENTS_PATH, lines=True)

comments["parent_submission_id"] = comments["parent_id"].map(normalize_parent_id)

submissions = submissions.sort_values("score", ascending=False)

# -------------------------
# Main loop
# -------------------------
for _, submission in tqdm(
    submissions.iterrows(),
    total=len(submissions),
    desc="Processing submissions"
):
    sid = submission["id"]

    sub_comments = comments[comments["parent_submission_id"] == sid]
    if sub_comments.empty:
        continue

    url = submission.get("url")
    if not isinstance(url, str):
        continue

    img_url = normalize_imgur_url(url)
    img_path = os.path.join(IMG_DIR, f"{sid}_image.jpg")

    if not download_image(img_url, img_path, sid):
        continue

    # Save comments sorted by score
    sub_comments = sub_comments.sort_values("score", ascending=False)

    comments_out = os.path.join(COMMENTS_DIR, f"{sid}_comments.jsonl")
    with open(comments_out, "w", encoding="utf-8") as f:
        for _, c in sub_comments.iterrows():
            f.write(json.dumps({
                "id": c["id"],
                "score": int(c["score"]),
                "body": c["body"],
            }, ensure_ascii=False) + "\n")

    # Save metadata
    meta = {
        "image_url": img_url,
        "submission_body": submission.get("selftext", ""),
        "submission_score": int(submission["score"]),
    }

    meta_out = os.path.join(META_DIR, f"{sid}_meta.json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
