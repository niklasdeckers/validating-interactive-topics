import json
from pathlib import Path
from collections import defaultdict

USERS_OF_INTEREST = ["thagen", "deckersn", "kgutekunst"]

def bucket_for_id(t, d, k):
    if t != d:
        return "mismatch"
    elif t == d == k:
        return "identical"
    else:
        return "switched"

# store: html_items[dataset][bucket] = list of image urls
html_items = defaultdict(lambda: defaultdict(list))

for dataset in ["lexica", "pexels", "reddit"]:

    # separate by file position
    labels_1_50 = {u: {} for u in USERS_OF_INTEREST}
    labels_51_100 = {u: {} for u in USERS_OF_INTEREST}

    # also store im_url per id (same for all users in a dataset)
    id_to_url = {}

    for path in Path(dataset).glob("*.jsonl"):
        username = path.stem
        if username not in USERS_OF_INTEREST:
            continue

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                obj = json.loads(line)
                _id = obj["id"]
                label = obj["label"][0]
                im_url = obj.get("im_url")

                # remember image url once
                if im_url is not None:
                    id_to_url[_id] = im_url

                if 1 <= i <= 50:
                    labels_1_50[username][_id] = label
                elif 51 <= i <= 100:
                    labels_51_100[username][_id] = label

    def process_range(labels_subset, tag):
        common_ids = set.intersection(
            *(set(labels_subset[u].keys()) for u in USERS_OF_INTEREST)
        )

        for _id in common_ids:
            t = labels_subset["thagen"][_id]
            d = labels_subset["deckersn"][_id]
            k = labels_subset["kgutekunst"][_id]

            bucket = bucket_for_id(t, d, k)

            if _id in id_to_url:
                html_items[f"{dataset} ({tag})"][bucket].append(
                    ( _id, id_to_url[_id], t, d, k )
                )

    # run for both splits
    process_range(labels_1_50, "entries 1–50")
    process_range(labels_51_100, "entries 51–100")

# ---------- BUILD HTML ----------
html = ["<html><head>",
        "<style>",
        "body{font-family:Arial;margin:20px;} ",
        ".grid{display:flex;flex-wrap:wrap;gap:10px;} ",
        ".item{width:200px;} ",
        "img{width:200px;border:1px solid #ccc;} ",
        ".meta{font-size:12px;color:#555;} ",
        "h2{margin-top:30px;} h3{margin-top:20px;} ",
        "</style>",
        "</head><body>",
        "<h1>Label Agreement Gallery</h1>"]

for dataset, buckets in html_items.items():
    html.append(f"<h2>{dataset}</h2>")

    for bucket in ["mismatch", "identical", "switched"]:
        items = buckets.get(bucket, [])
        html.append(f"<h3>{bucket} ({len(items)})</h3>")
        html.append("<div class='grid'>")

        for _id, url, t, d, k in items:
            html.append(
                f"<div class='item'>"
                f"<img src='{url}' loading='lazy'><br>"
                f"<div class='meta'>id: {_id}<br>"
                f"thagen={t}, deckersn={d}, kgutekunst={k}</div>"
                f"</div>"
            )

        html.append("</div>")

html.append("</body></html>")

out_path = Path("agreement_gallery.html")
out_path.write_text("\n".join(html), encoding="utf-8")

print(f"Written: {out_path.absolute()}")
