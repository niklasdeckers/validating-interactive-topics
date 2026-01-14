import json
import re
import shutil
import argparse
from pathlib import Path

# -------- CLI --------
parser = argparse.ArgumentParser()
parser.add_argument("--preview", action="store_true", help="Dry-run: do not modify files")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

PREVIEW = args.preview
VERBOSE = args.verbose

def log(msg):
    print(msg)

# -------- paths --------
OUTPUT_DIR = Path("output")
COMMENTS_DIR = OUTPUT_DIR / "comments"
IMAGES_DIR = OUTPUT_DIR / "images"
META_DIR = OUTPUT_DIR / "meta"
REMOVED_DIR = OUTPUT_DIR / "removed_by_blocklist"

if not PREVIEW:
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)

# -------- blocklist regexes --------
BLOCKLIST_PATTERNS = [
    ("deleted-only",
     r"^\s*\[\s*deleted\s*\]\s*$"),
     
    ("removed-only",
     r"^\s*\[\s*removed\s*\]\s*$"),
     
    ("caption-request",
     r"(?:^|\b)(?:caption\s+this|caption\s+(?!:).*?\bplease|please\b.*?\bcaption\b(?!\s*:))"),

    ("captionthis-literal",
     r"captionthis"),

    ("contest-no-winner",
     r"the\s+contest\s+has\s+concluded.*no\s+comment\s+got\s+more\s+than"),

    ("submission-success",
     r"your\s+submission\s+was\s+successful.*contest\s+will\s+conclude"),

    ("violation-message",
     r"sorry\s*,?\s*this\s+submission\s+violates"),

    ("markdown-link",
     r"\[[^\]]+\]\([^)]+\)"),

    ("http-link",
     r"https?://\S+"),

    ("winner-chosen",
     r"the\s+winner\s+has\s+been\s+choo?sen"),
     
    ("self-deleted-apology",
     r"i[^a-zA-Z]?\s*m\s+sorry\s*[-–—]?\s*i\s+deleted\s+it"),
]

BLOCKLIST_REGEXES = [
    (name, re.compile(pattern, flags=re.IGNORECASE | re.DOTALL))
    for name, pattern in BLOCKLIST_PATTERNS
]

def match_blocklist(text: str):
    for name, regex in BLOCKLIST_REGEXES:
        if regex.search(text):
            return name
    return None

# -------- main processing --------
for comments_file in COMMENTS_DIR.glob("*_comments.jsonl"):
    item_id = comments_file.name.replace("_comments.jsonl", "")
    kept_lines = []
    removed_lines = []

    if VERBOSE:
        log(f"\n▶ Processing {comments_file}")

    with comments_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue

            text = obj.get("body", "")
            rule = match_blocklist(text)

            if rule:
                removed_lines.append(line)
                if VERBOSE or PREVIEW:
                    snippet = text.replace("\n", " ")[:120]
                    log(f"  - REMOVE line {i} [{rule}]: {snippet!r}")
            else:
                kept_lines.append(line)

    # handle removed comments
    if removed_lines:
        target = REMOVED_DIR / comments_file.name
        if PREVIEW:
            log(f"  → Would append {len(removed_lines)} lines to {target}")
        else:
            with target.open("a", encoding="utf-8") as f:
                for line in removed_lines:
                    f.write(line)

    # handle original file
    if kept_lines:
        if not PREVIEW:
            with comments_file.open("w", encoding="utf-8") as f:
                for line in kept_lines:
                    f.write(line)
    else:
        if PREVIEW:
            log(f"  → Would delete empty {comments_file}")
        else:
            comments_file.unlink()

        # image
        image_path = IMAGES_DIR / f"{item_id}_image.jpg"
        if image_path.exists():
            if PREVIEW:
                log(f"  → Would move image {image_path} → {REMOVED_DIR}")
            else:
                shutil.move(str(image_path), str(REMOVED_DIR / image_path.name))

        # meta
        meta_path = META_DIR / f"{item_id}_meta.json"
        if meta_path.exists():
            if PREVIEW:
                log(f"  → Would move meta {meta_path} → {REMOVED_DIR}")
            else:
                shutil.move(str(meta_path), str(REMOVED_DIR / meta_path.name))

if PREVIEW:
    log("\n✔ Preview complete — no files were modified.")