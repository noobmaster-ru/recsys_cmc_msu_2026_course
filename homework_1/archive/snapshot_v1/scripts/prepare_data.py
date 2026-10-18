#!/usr/bin/env python3
"""Download, clean and audit Amazon Video Games 2018; standard library only."""

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import math
import re
import statistics
import urllib.request
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/"
SOURCES = {
    "Video_Games_5.json.gz": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz",
        "sha256": "a0bd63e90ad3bd577778678672bb2e7e88b8c085683c86aca2f18c1ab543cb84",
    },
    "meta_Video_Games.json.gz": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz",
        "sha256": "4069e17c7efa33e7574c2e77f1fe0b0c6f2a42a7ac29c4422696d788d8a8e0a6",
    },
}
CATEGORY_BOUNDARY = re.compile(r"(?:</span>\s*){3}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obtain(path, source, download):
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path}: rerun with --download")
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {path.name}", flush=True)
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            expected_size = response.headers.get("Content-Length")
            size = 0
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
        if expected_size is not None and size != int(expected_size):
            raise ValueError(f"Incomplete download: {path.name}")
        temporary.replace(path)
    digest = sha256(path)
    if source["sha256"] and digest != source["sha256"]:
        raise ValueError(f"Source SHA-256 mismatch: {path.name}")
    return {"url": source["url"], "bytes": path.stat().st_size, "sha256": digest}


def read_jsonl(path):
    # Reading through EOF also checks the gzip CRC and catches truncated archives.
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, TypeError) as error:
                raise ValueError(f"Malformed JSON at {path.name}:{line_number}") from error
            yield record


def clean_text(value):
    if not isinstance(value, str):
        return ""
    value = html.unescape(html.unescape(value))
    value = re.sub(r"<[^>]*>", " ", value)
    return " ".join(value.split()).strip()


def clean_list(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        for text in clean_list(entry) if isinstance(entry, list) else [clean_text(entry)]:
            if text and text not in result:
                result.append(text)
    return result


def clean_categories(value):
    # Source scraper sometimes appends product-description bullets AFTER the
    # closing breadcrumb spans. Those strings are not taxonomy categories.
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        parts = CATEGORY_BOUNDARY.split(html.unescape(entry), maxsplit=1)
        text = clean_text(parts[0])
        if text and text not in result:
            result.append(text)
        if len(parts) > 1:
            break
    return result


def normalize_item(record):
    brand = re.sub(r"^by\s+", "", clean_text(record.get("brand")))
    return {
        "item_id": clean_text(record.get("asin")),
        "title": clean_text(record.get("title")),
        "brand": brand,
        "categories": clean_categories(record.get("category", record.get("categories"))),
        "description": clean_list(record.get("description")),
        "features": clean_list(record.get("feature")),
    }


def metadata_score(record):
    return (
        bool(record["title"]), bool(record["categories"]), bool(record["brand"]),
        len(record["description"]) + len(record["features"]), len(record["title"]),
    )


def snapshot(rows):
    return {
        "interactions": len(rows), "users": len({r[0] for r in rows}),
        "items": len({r[1] for r in rows}),
    }


def degree_summary(counts):
    values = list(counts.values())
    return {
        "min": min(values), "median": statistics.median(values),
        "mean": statistics.mean(values), "max": max(values),
    }


@contextmanager
def deterministic_gzip(path):
    # Fix gzip header time/name so identical inputs produce identical file hashes.
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                yield text
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing raw archives")
    args = parser.parse_args()
    raw, processed = ROOT / "data/raw", ROOT / "data/processed"
    for folder in (raw, processed, ROOT / "reports"):
        folder.mkdir(parents=True, exist_ok=True)
    sources = {name: obtain(raw / name, source, args.download) for name, source in SOURCES.items()}
    report = {"dataset": "Amazon Reviews 2018 / Video Games / 5-core", "source_page": SOURCE_PAGE,
              "sources": sources, "cleaning_version": 1, "stages": []}

    print("Auditing interaction records", flush=True)
    raw_count, invalid = 0, Counter()
    raw_users, raw_items, signatures = set(), set(), set()
    full_duplicates, repeated_pairs, conflicting_pairs = 0, set(), set()
    pair_counts, pairs = Counter(), {}
    for record in read_jsonl(raw / "Video_Games_5.json.gz"):
        raw_count += 1
        signature = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).digest()
        if signature in signatures:
            full_duplicates += 1
        signatures.add(signature)
        user, item = record.get("reviewerID"), record.get("asin")
        rating, timestamp = record.get("overall"), record.get("unixReviewTime")
        if isinstance(user, str) and user.strip():
            raw_users.add(user.strip())
        if isinstance(item, str) and item.strip():
            raw_items.add(item.strip())
        reasons = []
        if not isinstance(user, str) or not user.strip():
            reasons.append("missing_user_id")
        if not isinstance(item, str) or not item.strip():
            reasons.append("missing_item_id")
        if (isinstance(rating, bool) or not isinstance(rating, (int, float))
                or not math.isfinite(rating) or rating not in (1, 2, 3, 4, 5)):
            reasons.append("invalid_rating")
        # The source covers reviews through 2018; the upper bound is fixed, not today's date.
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or not 0 < timestamp < 1546300800:
            reasons.append("invalid_timestamp")
        if reasons:
            invalid.update(reasons)
            invalid["records"] += 1
            continue
        row = (user.strip(), item.strip(), int(rating), timestamp)
        pair = row[:2]
        pair_counts[pair] += 1
        if pair in pairs:
            repeated_pairs.add(pair)
            if pairs[pair][2] != row[2]:
                conflicting_pairs.add(pair)
            # First occurrence in chronological time; original line order breaks ties.
            if row[3] < pairs[pair][3]:
                pairs[pair] = row
        else:
            pairs[pair] = row
    report["raw_interactions"] = {"interactions": raw_count, "users": len(raw_users), "items": len(raw_items)}
    report["interaction_quality"] = {
        "invalid": dict(sorted(invalid.items())), "exact_duplicate_records": full_duplicates,
        "repeated_user_item_pairs": len(repeated_pairs),
        "pairs_with_conflicting_ratings": len(conflicting_pairs),
        "removed_repeated_pair_records": sum(pair_counts.values()) - len(pairs),
    }
    report["stages"].append({"stage": "valid_records", "interactions": sum(pair_counts.values()),
                             "users": len({p[0] for p in pairs}), "items": len({p[1] for p in pairs})})
    rows = list(pairs.values())
    report["stages"].append({"stage": "earliest_record_per_user_item", **snapshot(rows)})
    del signatures, pair_counts, pairs

    print("Auditing and normalizing item metadata", flush=True)
    items, metadata_rows, invalid_item_ids, duplicate_item_rows, changed_rows = {}, 0, 0, 0, 0
    category_tail_records = 0
    for record in read_jsonl(raw / "meta_Video_Games.json.gz"):
        metadata_rows += 1
        categories = record.get("category", [])
        if isinstance(categories, list) and any(
                isinstance(c, str) and CATEGORY_BOUNDARY.search(html.unescape(c)) for c in categories):
            category_tail_records += 1
        item = normalize_item(record)
        if not item["item_id"]:
            invalid_item_ids += 1
            continue
        if any(item[k] != record.get(original) for k, original in
               (("title", "title"), ("brand", "brand"), ("categories", "category"))):
            changed_rows += 1
        key = item["item_id"]
        if key in items:
            duplicate_item_rows += 1
        if key not in items or metadata_score(item) > metadata_score(items[key]):
            items[key] = item
    observed = {r[1] for r in rows}
    without_metadata = observed - items.keys()
    without_title = {i for i in observed & items.keys() if not items[i]["title"]}
    report["metadata_quality"] = {
        "raw_records": metadata_rows, "unique_item_ids": len(items),
        "invalid_item_ids": invalid_item_ids, "duplicate_item_records": duplicate_item_rows,
        "records_with_normalized_title_brand_or_categories": changed_rows,
        "records_with_category_breadcrumb_boundary": category_tail_records,
        "observed_items_without_metadata": len(without_metadata),
        "observed_items_without_title": len(without_title),
    }
    usable_items = {i for i in observed & items.keys() if items[i]["title"]}
    rows = [r for r in rows if r[1] in usable_items]
    report["stages"].append({"stage": "join_metadata_with_nonempty_title", **snapshot(rows)})
    iterations = []
    while True:
        users, products = Counter(r[0] for r in rows), Counter(r[1] for r in rows)
        filtered = [r for r in rows if users[r[0]] >= 5 and products[r[1]] >= 5]
        if len(filtered) == len(rows):
            break
        iterations.append({"removed_interactions": len(rows) - len(filtered), **snapshot(filtered)})
        rows = filtered
    if not rows:
        raise ValueError("Cleaning produced an empty dataset")
    rows.sort(key=lambda r: (r[3], r[0], r[1]))
    products, users = Counter(r[1] for r in rows), Counter(r[0] for r in rows)
    item_ids = sorted(products)
    report["five_core_iterations"] = iterations
    report["stages"].append({"stage": "restore_five_core", **snapshot(rows)})
    final = snapshot(rows)
    final.update({
        "rating_counts": dict(sorted(Counter(str(r[2]) for r in rows).items())),
        "start_utc": datetime.fromtimestamp(rows[0][3], timezone.utc).isoformat(),
        "end_utc": datetime.fromtimestamp(rows[-1][3], timezone.utc).isoformat(),
        "density": len(rows) / (len(users) * len(products)),
        "interactions_per_user": degree_summary(users), "interactions_per_item": degree_summary(products),
        "positive_ratings_at_least_four": snapshot([r for r in rows if r[2] >= 4]),
    })
    report["cleaned"] = final
    report["metadata_coverage"] = {
        field: {"items_with_value": sum(bool(items[i][field]) for i in item_ids),
                "fraction": sum(bool(items[i][field]) for i in item_ids) / len(item_ids)}
        for field in ("title", "brand", "categories", "description", "features")
    }
    report["metadata_coverage"]["unique_brands"] = len({items[i]["brand"] for i in item_ids if items[i]["brand"]})
    report["metadata_coverage"]["unique_categories"] = len({c for i in item_ids for c in items[i]["categories"]})
    report["metadata_coverage"]["items_with_category_other_than_video_games"] = sum(
        any(c != "Video Games" for c in items[i]["categories"]) for i in item_ids)
    checks = {
        "interaction_count_100000_to_2000000": 100000 <= len(rows) <= 2000000,
        "at_least_1000_users": len(users) >= 1000,
        "at_least_1000_items": len(products) >= 1000,
        "unique_user_item_pairs": len({r[:2] for r in rows}) == len(rows),
        "all_ratings_in_1_to_5": all(1 <= r[2] <= 5 for r in rows),
        "all_items_have_title_and_metadata": all(items[i]["title"] for i in item_ids),
        "at_least_1000_items_with_non_id_metafeatures": sum(
            bool(items[i]["brand"] or items[i]["categories"] or items[i]["description"]) for i in item_ids) >= 1000,
        "minimum_five_interactions_per_user_and_item": min(users.values()) >= 5 and min(products.values()) >= 5,
    }
    report["checks"] = checks
    report["all_data_checks_passed"] = all(checks.values())
    report["policies"] = {
        "interaction": "One earliest observed rating per user-item; same-time ties use source line order.",
        "metadata": "Normalize HTML/whitespace; choose most informative duplicate metadata row; require title.",
        "five_core": "Iteratively filter users and items with fewer than five remaining interactions.",
        "sampling": "Full supplied category 5-core; no random subsampling and no random operations.",
        "positive_relevance": "rating >= 4 is a diagnostic only; no ratings removed by this threshold.",
        "temporal_features": "Metadata are a crawl-time snapshot, not historical versions. Static catalog availability is an assumption. No price/rank/related-item links, votes or review texts exported as features.",
        "selection_bias": "The supplied and restored 5-core use the full observation period. This is cohort selection, not a production cold-start evaluation; train-only filtering must be considered in the evaluation protocol.",
        "pending": "Seminar dataset exclusion and the train/validation/test protocol require separate verification.",
    }
    print("Writing cleaned tables", flush=True)
    with deterministic_gzip(processed / "interactions.csv.gz") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "item_id", "rating", "timestamp"))
        writer.writerows(rows)
    with deterministic_gzip(processed / "items.jsonl.gz") as stream:
        for item_id in item_ids:
            stream.write(json.dumps(items[item_id], ensure_ascii=False, sort_keys=True) + "\n")
    report["outputs"] = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                         for path in sorted(processed.glob("*.gz")) if path.name in ("interactions.csv.gz", "items.jsonl.gz")}
    audit_path = ROOT / "reports/data_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cleaned": final, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    if not report["all_data_checks_passed"]:
        raise SystemExit("Dataset does not meet all data checks; inspect reports/data_audit.json")


if __name__ == "__main__":
    main()
