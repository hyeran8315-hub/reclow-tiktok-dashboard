"""Weekly Reclow TikTok data updater.

1. Runs the saved Apify TikTok Scraper task.
2. Downloads the run's default dataset.
3. Cleans and analyzes comments.
4. Merges them with the existing dashboard CSV.

Required environment variables:
    APIFY_TOKEN
Optional environment variables:
    APIFY_TASK_ID (default: respectable_tabla_b2q~reclow-weekly-tiktok-scraper)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from langdetect import DetectorFactory, LangDetectException, detect
from transformers import pipeline

API_BASE = "https://api.apify.com/v2"
TASK_ID = os.getenv(
    "APIFY_TASK_ID",
    "respectable_tabla_b2q~reclow-weekly-tiktok-scraper",
)
TOKEN = os.getenv("APIFY_TOKEN", "").strip()
OUTPUT_PATH = Path("reclow_tiktok_analysis.csv")
POLL_INTERVAL_SECONDS = 15
RUN_TIMEOUT_SECONDS = 45 * 60

REQUIRED_COLUMNS = [
    "text",
    "diggCount",
    "uniqueId",
    "createTimeISO",
    "videoWebUrl",
]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["language", "sentiment", "category"]

DetectorFactory.seed = 42


def api_request(method: str, endpoint: str, **kwargs: Any) -> requests.Response:
    """Call Apify API and raise a readable error on failure."""
    params = dict(kwargs.pop("params", {}) or {})
    params["token"] = TOKEN
    response = requests.request(
        method,
        f"{API_BASE}{endpoint}",
        params=params,
        timeout=60,
        **kwargs,
    )
    response.raise_for_status()
    return response


def run_task_and_get_dataset_id() -> str:
    """Start the saved task, wait for completion, return dataset ID."""
    if not TOKEN:
        raise RuntimeError("APIFY_TOKEN environment variable is missing.")

    print(f"Starting Apify task: {TASK_ID}")
    response = api_request("POST", f"/actor-tasks/{TASK_ID}/runs")
    run = response.json()["data"]
    run_id = run["id"]
    print(f"Apify run started: {run_id}")

    deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        run = api_request("GET", f"/actor-runs/{run_id}").json()["data"]
        status = run.get("status", "UNKNOWN")
        print(f"Run status: {status}")

        if status == "SUCCEEDED":
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                raise RuntimeError("Run succeeded but no default dataset was returned.")
            return dataset_id

        if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
            message = run.get("statusMessage") or "No status message"
            raise RuntimeError(f"Apify run ended with {status}: {message}")

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError("Apify run did not finish within 45 minutes.")


def download_dataset(dataset_id: str) -> list[dict[str, Any]]:
    """Download all items from the run's default dataset."""
    print(f"Downloading dataset: {dataset_id}")
    response = api_request(
        "GET",
        f"/datasets/{dataset_id}/items",
        params={"clean": "true", "format": "json"},
    )
    items = response.json()
    if not isinstance(items, list):
        raise RuntimeError("Unexpected dataset response format.")
    print(f"Downloaded items: {len(items)}")
    return items


def first_present(row: pd.Series, candidates: list[str], default: Any = "") -> Any:
    for name in candidates:
        if name in row.index:
            value = row[name]
            if pd.notna(value) and str(value).strip() != "":
                return value
    return default


def normalize_raw_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Map possible scraper output field names to dashboard columns."""
    raw = pd.json_normalize(items)
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    records: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        text = first_present(row, ["text", "commentText", "comment", "description"])
        # Ignore post-only items that do not look like comments.
        if not str(text).strip():
            continue

        records.append(
            {
                "text": text,
                "diggCount": first_present(
                    row,
                    ["diggCount", "likes", "likeCount", "commentLikeCount"],
                    0,
                ),
                "uniqueId": first_present(
                    row,
                    [
                        "uniqueId",
                        "authorMeta.name",
                        "author.uniqueId",
                        "user.uniqueId",
                        "detailedMentions/0/uniqueId",
                    ],
                    "unknown",
                ),
                "createTimeISO": first_present(
                    row,
                    ["createTimeISO", "createTime", "createdAt"],
                    pd.NaT,
                ),
                "videoWebUrl": first_present(
                    row,
                    ["videoWebUrl", "webVideoUrl", "url", "videoUrl"],
                    "",
                ),
            }
        )

    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned["uniqueId"] = cleaned["uniqueId"].fillna("unknown").astype(str)
    cleaned["videoWebUrl"] = cleaned["videoWebUrl"].fillna("").astype(str)
    cleaned["diggCount"] = pd.to_numeric(cleaned["diggCount"], errors="coerce").fillna(0).astype(int)
    cleaned["createTimeISO"] = pd.to_datetime(cleaned["createTimeISO"], errors="coerce", utc=True)
    cleaned = cleaned.drop_duplicates(
        subset=["text", "uniqueId", "videoWebUrl"], keep="last"
    ).reset_index(drop=True)
    return cleaned


def detect_language(text: str) -> str:
    text = str(text).strip()
    if len(text) < 3:
        return "unknown"
    try:
        lang = detect(text)
    except LangDetectException:
        return "unknown"

    if lang == "en":
        return "English"
    if lang in {"ms", "id"}:
        return "Malay/Indonesian"
    return "Other"


def classify_category(text: str) -> str:
    value = str(text).lower()
    if re.search(r"price|expensive|cheap|cost|\brm\b|discount|sale|promo", value):
        return "Price"
    if re.search(r"design|style|frame|look|beautiful|cute|cool|fashion", value):
        return "Design"
    if re.search(r"quality|durable|material|comfortable|good|bad", value):
        return "Quality"
    if re.search(r"delivery|shipping|arrive|parcel|receive|received", value):
        return "Delivery"
    if re.search(r"buy|order|purchase|want|need|get one|take my money", value):
        return "Purchase Intention"
    if re.search(r"where|how|when|available|stock|shop|location|link", value):
        return "Inquiry"
    return "Other"


def add_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    analyzed = df.copy()
    analyzed["language"] = analyzed["text"].apply(detect_language)
    analyzed["category"] = analyzed["text"].apply(classify_category)

    print("Loading multilingual sentiment model...")
    sentiment_model = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=-1,
    )
    label_map = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
    }

    texts = analyzed["text"].astype(str).str.slice(0, 512).tolist()
    print(f"Analyzing sentiment for {len(texts)} comments...")
    results = sentiment_model(texts, truncation=True, batch_size=16)
    analyzed["sentiment"] = [
        label_map.get(str(result.get("label", "neutral")), "Neutral")
        for result in results
    ]
    return analyzed[OUTPUT_COLUMNS]


def load_existing() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    existing = pd.read_csv(OUTPUT_PATH, low_memory=False)
    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
    return existing[OUTPUT_COLUMNS]


def merge_and_save(new_data: pd.DataFrame) -> None:
    existing = load_existing()
    combined = pd.concat([existing, new_data], ignore_index=True)

    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined["uniqueId"] = combined["uniqueId"].fillna("unknown").astype(str)
    combined["videoWebUrl"] = combined["videoWebUrl"].fillna("").astype(str)
    combined["diggCount"] = pd.to_numeric(combined["diggCount"], errors="coerce").fillna(0).astype(int)
    combined["createTimeISO"] = pd.to_datetime(
        combined["createTimeISO"], errors="coerce", utc=True
    )

    combined = combined[combined["text"] != ""]
    combined = combined.drop_duplicates(
        subset=["text", "uniqueId", "videoWebUrl"], keep="last"
    )
    combined = combined.sort_values("createTimeISO", ascending=False, na_position="last")
    combined["createTimeISO"] = combined["createTimeISO"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    combined = combined[OUTPUT_COLUMNS].reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(combined)} total comments to {OUTPUT_PATH}")


def main() -> int:
    try:
        dataset_id = run_task_and_get_dataset_id()
        items = download_dataset(dataset_id)
        raw_comments = normalize_raw_items(items)
        cleaned = clean_comments(raw_comments)
        print(f"Valid comments in this run: {len(cleaned)}")
        if cleaned.empty:
            raise RuntimeError(
                "No valid comment rows were found. Check the Apify task's TikTok comments settings."
            )
        analyzed = add_analysis(cleaned)
        merge_and_save(analyzed)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
