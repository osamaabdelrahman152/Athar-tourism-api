"""
Recommendation Model — Travel Itinerary Generator
Converts places_all_v1.json into an optimized day-by-day itinerary.

Usage:
    python model.py --data places_all_v1.json \
                    --governorate Luxor \
                    --budget 3000 \
                    --days 3 \
                    --interests historical pharaonic temple museum
"""

# =============================================================================
# Step 1 — Setup & Imports
# =============================================================================
import json
import re
import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print(" Libraries loaded successfully")


# =============================================================================
# Step 2 — Preprocessing & Feature Engineering
# =============================================================================

GOVERNORATE_MAP = {
    # Cairo
    "cairo":         "Cairo",
    "islamic cairo": "Cairo",
    "coptic cairo":  "Cairo",
    "heliopolis":    "Cairo",
    "manial":        "Cairo",
    "agouza":        "Cairo",
    "dokki":         "Cairo",

    # Giza
    "giza":          "Giza",
    "haram":         "Giza",
    "6th october":   "Giza",
    "sheikh zayed":  "Giza",
    "saqqara":       "Giza",
    "dahshur":       "Giza",
    "badrashin":     "Giza",
    "fayoum":        "Giza",

    # Luxor
    "luxor":         "Luxor",
    "west bank":     "Luxor",
    "north luxor":   "Luxor",
    "sohag":         "Luxor",

    # Aswan
    "aswan":         "Aswan",
}


def extract_governorate(location_str: str) -> str:
    loc_lower = location_str.lower()
    for keyword, gov in GOVERNORATE_MAP.items():
        if keyword in loc_lower:
            return gov
    return "Unknown"


def parse_duration(duration_str: str) -> float:
    """
    '2–3 hours'  → 2.5
    '1 hour'     → 1.0
    '30 minutes' → 0.5
    """
    if not isinstance(duration_str, str):
        return 2.0

    duration_str = duration_str.lower().replace("–", "-").replace("—", "-")

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", duration_str)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        hours = (low + high) / 2
        if "minute" in duration_str:
            hours /= 60
        return round(hours, 2)

    single_match = re.search(r"(\d+(?:\.\d+)?)", duration_str)
    if single_match:
        val = float(single_match.group(1))
        if "minute" in duration_str:
            return round(val / 60, 2)
        return round(val, 2)

    return 2.0


def extract_tags(tags_val) -> list:
    """
    Input:  ["historical / تاريخي", "pharaonic / فرعوني"]
    Output: ["historical", "pharaonic"]
    """
    if not isinstance(tags_val, list):
        return []
    clean = []
    for tag in tags_val:
        en_part = tag.split("/")[0].strip().lower()
        en_part = re.sub(r"[^a-z0-9 ]", "", en_part).strip()
        if en_part:
            clean.append(en_part)
    return clean


def load_and_preprocess(data_path: str) -> pd.DataFrame:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f" Loaded {len(raw_data)} places from '{data_path}'")

    df = pd.DataFrame(raw_data)

    df["governorate"]       = df["location"].apply(extract_governorate)
    df["avg_duration_hours"] = df["recommended_duration"].apply(parse_duration)
    df["tags_clean"]         = df["tags"].apply(extract_tags)

    all_tags = set()
    for tags in df["tags_clean"]:
        all_tags.update(tags)

    print(f" Step 2 complete — {len(df)} places, {len(all_tags)} unique tags")
    return df


# =============================================================================
# Step 3 — Scoring Engine
# =============================================================================

def compute_score(row, interests: list, daily_budget: float) -> float:
    """
    Composite score (0–1) based on:
      - Tag Match Score  (50%): How many user interests match place tags
      - Budget Fit Score (25%): Cheaper places score higher
      - Popularity Score (25%): Normalized popularity_score from dataset
    """
    # Tag Match
    if not interests or not row["tags_clean"]:
        tag_score = 0.0
    else:
        matched = len(set(interests) & set(row["tags_clean"]))
        tag_score = matched / len(interests)

    # Budget Fit
    if daily_budget > 0:
        budget_fit = 1.0 - (row["price_egp"] / (daily_budget * 0.6))
        budget_fit = max(0.0, min(1.0, budget_fit))
    else:
        budget_fit = 1.0

    # Popularity
    pop_score = row["popularity_score"] / 100.0

    return round(0.50 * tag_score + 0.25 * budget_fit + 0.25 * pop_score, 4)


def score_and_filter(df: pd.DataFrame, user_input: dict) -> pd.DataFrame:
    governorate  = user_input["governorate"]
    budget_egp   = user_input["budget_egp"]
    duration_days = user_input["duration_days"]
    interests    = user_input["interests"]

    daily_budget = budget_egp / duration_days
    budget_cap   = daily_budget * 0.6

    df_filtered = df[df["governorate"] == governorate].copy()
    print(f"\n Places in {governorate}: {len(df_filtered)}")

    df_filtered = df_filtered[df_filtered["price_egp"] <= budget_cap].copy()
    print(f" Daily budget: {daily_budget:.0f} EGP | Price cap: {budget_cap:.0f} EGP")
    print(f"   Places within budget: {len(df_filtered)}")

    df_filtered["score"] = df_filtered.apply(
        lambda row: compute_score(row, interests, daily_budget), axis=1
    )

    max_places = duration_days * 4
    df_scored = (
        df_filtered
        .sort_values("score", ascending=False)
        .head(max_places)
        .copy()
    )

    print(f"\n Top {max_places} scored places selected")
    return df_scored, daily_budget


# =============================================================================
# Step 4 — K-Means Clustering (Geographic Day Grouping)
# =============================================================================

def cluster_by_geography(df_scored: pd.DataFrame, n_days: int):
    coords        = df_scored[["latitude", "longitude"]].values
    scaler        = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    kmeans = KMeans(
        n_clusters=n_days,
        init="k-means++",
        n_init=20,
        random_state=42,
    )

    df_scored["day_cluster"] = kmeans.fit_predict(coords_scaled)

    # Sort clusters north → south (higher latitude = earlier day)
    cluster_centers   = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_lat_order = cluster_centers[:, 0].argsort()[::-1]
    cluster_to_day    = {
        cluster_id: day_num + 1
        for day_num, cluster_id in enumerate(cluster_lat_order)
    }

    df_scored["day"] = df_scored["day_cluster"].map(cluster_to_day)
    df_scored = df_scored.sort_values(
        ["day", "score"], ascending=[True, False]
    ).reset_index(drop=True)

    return df_scored, kmeans, scaler


# =============================================================================
# Step 5 — Constraint Optimization & JSON Output
# =============================================================================

MAX_HOURS_PER_DAY = 8.0


def optimize_itinerary(df_candidates: pd.DataFrame, n_days: int,
                       max_hours: float, max_budget: float) -> dict:
    """
    Greedy redistribution of places across days respecting hour and budget caps.
    Higher-score places are assigned first; each place goes to its preferred
    cluster day, falling back to other days if needed.
    """
    df_sorted = df_candidates.sort_values(
        ["day_cluster", "score"], ascending=[True, False]
    ).reset_index(drop=True)

    days = {
        i: {"places": [], "hours": 0.0, "cost": 0.0}
        for i in range(1, n_days + 1)
    }

    for _, place in df_sorted.iterrows():
        preferred_day = place["day"]
        day_order = [preferred_day] + [d for d in range(1, n_days + 1) if d != preferred_day]

        for day in day_order:
            new_hours = days[day]["hours"] + place["avg_duration_hours"]
            new_cost  = days[day]["cost"]  + place["price_egp"]

            if new_hours <= max_hours and new_cost <= max_budget:
                days[day]["places"].append(place)
                days[day]["hours"] += place["avg_duration_hours"]
                days[day]["cost"]  += place["price_egp"]
                break

    return days


def build_itinerary_json(optimized: dict, user_input: dict, n_days: int) -> dict:
    total_cost   = sum(d["cost"]         for d in optimized.values())
    total_hours  = sum(d["hours"]        for d in optimized.values())
    total_places = sum(len(d["places"])  for d in optimized.values())

    itinerary_json = {
        "trip_summary": {
            "governorate":      user_input["governorate"],
            "duration_days":    user_input["duration_days"],
            "total_budget_egp": user_input["budget_egp"],
            "total_cost_egp":   round(total_cost, 2),
            "budget_used_pct":  round(total_cost / user_input["budget_egp"] * 100, 1),
            "total_places":     total_places,
            "total_hours":      round(total_hours, 1),
            "interests":        user_input["interests"],
        },
        "itinerary": [],
    }

    for day, data in optimized.items():
        day_entry = {
            "day":            day,
            "total_cost_egp": round(data["cost"], 2),
            "total_hours":    round(data["hours"], 1),
            "places":         [],
        }
        for p in data["places"]:
            day_entry["places"].append({
                "id":                 int(p["id"]),
                "name_ar":            p["name_ar"],
                "name_en":            p["name_en"],
                "category":           p["category"],
                "price_egp":          float(p["price_egp"]),
                "avg_duration_hours": float(p["avg_duration_hours"]),
                "rating":             float(p["rating"]),
                "score":              float(p["score"]),
                "latitude":           float(p["latitude"]),
                "longitude":          float(p["longitude"]),
                "tags":               p["tags_clean"],
                "opening_hours":      p["opening_hours"],
                "image_url":          p["image_url"],
            })
        itinerary_json["itinerary"].append(day_entry)

    return itinerary_json


# =============================================================================
# Step 6 — Validation & Quality Report
# =============================================================================

RULES = {
    "max_hours_per_day":   MAX_HOURS_PER_DAY,
    "min_places_per_day":  1,
    "max_places_per_day":  6,
    "min_budget_used_pct": 30.0,
    "min_avg_score":       0.50,
}


def validate_itinerary(itinerary_json: dict, max_budget_per_day: float) -> bool:
    rules   = {**RULES, "max_budget_per_day": max_budget_per_day}
    errors  = []
    warnings_list = []
    passed  = []

    summary = itinerary_json["trip_summary"]
    days    = itinerary_json["itinerary"]

    # ① JSON Structure
    required_summary = ["governorate", "duration_days", "total_budget_egp",
                        "total_cost_egp", "budget_used_pct", "total_places",
                        "total_hours", "interests"]
    required_place   = ["id", "name_ar", "name_en", "category", "price_egp",
                        "avg_duration_hours", "rating", "score",
                        "latitude", "longitude", "tags", "opening_hours", "image_url"]

    missing_summary = [f for f in required_summary if f not in summary]
    if missing_summary:
        errors.append(f" Missing summary fields: {missing_summary}")
    else:
        passed.append(" JSON structure — all required fields present")

    for day_data in days:
        for place in day_data["places"]:
            missing_place = [f for f in required_place if f not in place]
            if missing_place:
                errors.append(f" Place '{place.get('name_en', '?')}' missing: {missing_place}")
    if not any("missing" in e for e in errors):
        passed.append(" Place fields — all required fields present in every place")

    # ② Per-Day Constraints
    for day_data in days:
        d = day_data["day"]
        h = day_data["total_hours"]
        c = day_data["total_cost_egp"]
        n = len(day_data["places"])

        if h > rules["max_hours_per_day"]:
            errors.append(f" Day {d}: {h}h exceeds max {rules['max_hours_per_day']}h")
        else:
            passed.append(f" Day {d}: hours OK ({h}h ≤ {rules['max_hours_per_day']}h)")

        if c > rules["max_budget_per_day"]:
            errors.append(f" Day {d}: {c} EGP exceeds daily budget {rules['max_budget_per_day']} EGP")
        else:
            passed.append(f" Day {d}: budget OK ({c:.0f} EGP ≤ {rules['max_budget_per_day']:.0f} EGP)")

        if n < rules["min_places_per_day"]:
            errors.append(f" Day {d}: only {n} place(s) — too few")
        elif n > rules["max_places_per_day"]:
            warnings_list.append(f"  Day {d}: {n} places — consider reducing for comfort")
        else:
            passed.append(f" Day {d}: place count OK ({n} places)")

    # ③ Budget Utilization
    budget_pct = summary["budget_used_pct"]
    if budget_pct < rules["min_budget_used_pct"]:
        warnings_list.append(f"  Budget utilization low: {budget_pct}% — model may be too conservative")
    elif budget_pct > 100:
        errors.append(f" Budget exceeded: {budget_pct}%")
    else:
        passed.append(f" Budget utilization: {budget_pct}% (within range)")

    # ④ Score Quality
    all_scores = [p["score"] for d in days for p in d["places"]]
    avg_score  = sum(all_scores) / len(all_scores) if all_scores else 0
    if avg_score < rules["min_avg_score"]:
        warnings_list.append(f" Average score low: {avg_score:.3f}")
    else:
        passed.append(f" Recommendation quality: avg score {avg_score:.3f} (≥ {rules['min_avg_score']})")

    # ⑤ Coordinate Sanity
    for day_data in days:
        for place in day_data["places"]:
            lat, lng = place["latitude"], place["longitude"]
            if not (22.0 <= lat <= 32.0 and 25.0 <= lng <= 37.0):
                errors.append(f" '{place['name_en']}': coordinates out of Egypt bounds ({lat}, {lng})")
    if not any("coordinates" in e for e in errors):
        passed.append(" Coordinates — all places within Egypt bounds")

    # ⑥ Duplicate Places
    all_ids = [p["id"] for d in days for p in d["places"]]
    if len(all_ids) != len(set(all_ids)):
        errors.append(" Duplicate places detected in itinerary")
    else:
        passed.append(" No duplicate places")

    # Print Report
    print("=" * 65)
    print("            VALIDATION & QUALITY REPORT")
    print("=" * 65)
    print(f"\n PASSED ({len(passed)})")
    for p in passed:
        print(f"   {p}")
    if warnings_list:
        print(f"\n WARNINGS ({len(warnings_list)})")
        for w in warnings_list:
            print(f"   {w}")
    if errors:
        print(f"\n ERRORS ({len(errors)})")
        for e in errors:
            print(f"   {e}")
    else:
        print("\n NO ERRORS FOUND")

    status = " PASSED" if not errors else " FAILED"
    print("\n" + "=" * 65)
    print(f"  VALIDATION STATUS  :  {status}")
    print(f"  Governorate        :  {summary['governorate']}")
    print(f"  Duration           :  {summary['duration_days']} days")
    print(f"  Total Places       :  {summary['total_places']}")
    print(f"  Total Cost         :  {summary['total_cost_egp']:.0f} / {summary['total_budget_egp']} EGP")
    print(f"  Budget Used        :  {summary['budget_used_pct']}%")
    print(f"  Total Hours        :  {summary['total_hours']}h")
    print(f"  Avg Rec. Score     :  {avg_score:.3f}")
    print(f"  Checks Passed      :  {len(passed)}")
    print(f"  Warnings           :  {len(warnings_list)}")
    print(f"  Errors             :  {len(errors)}")
    print("=" * 65)

    return len(errors) == 0


# =============================================================================
# Step 7 — Save Model Artifacts
# =============================================================================

def save_artifacts(kmeans, scaler, n_days: int, user_input: dict,
                   max_hours: float, max_budget_per_day: float):
    model_artifacts = {
        "kmeans":                  kmeans,
        "scaler":                  scaler,
        "n_days":                  n_days,
        "trained_on_governorate":  user_input["governorate"],
    }
    with open("kmeans_model.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)
    print(" Saved: kmeans_model.pkl")

    pipeline_config = {
        "scoring_weights": {
            "tag_match":  0.50,
            "budget_fit": 0.25,
            "popularity": 0.25,
        },
        "constraints": {
            "max_hours_per_day":   max_hours,
            "budget_cap_ratio":    0.60,
            "max_places_per_trip": "duration_days * 4",
        },
        "validation_rules": {
            "max_hours_per_day":   max_hours,
            "max_budget_per_day":  max_budget_per_day,
            "min_places_per_day":  1,
            "max_places_per_day":  6,
            "min_budget_used_pct": 30.0,
            "min_avg_score":       0.50,
        },
        "supported_governorates": ["Cairo", "Giza", "Luxor", "Aswan"],
        "data_file":              "places_all_v1.json",
    }
    with open("pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, ensure_ascii=False, indent=2)
    print(" Saved: pipeline_config.json")

    # Verify
    with open("kmeans_model.pkl", "rb") as f:
        loaded = pickle.load(f)
    print("\n🔍 Verification:")
    print(f"   KMeans clusters : {loaded['kmeans'].n_clusters}")
    print(f"   Scaler mean     : {loaded['scaler'].mean_.round(4)}")
    print(f"   Trained on      : {loaded['trained_on_governorate']}")
    print(f"   n_days          : {loaded['n_days']}")
    print("\n Model artifacts saved")


# =============================================================================
# Main Pipeline
# =============================================================================

def pipeline_core(data_path: str, user_input: dict, save: bool = True) -> dict:
    # 1. Load & preprocess
    df = load_and_preprocess(data_path)

    # 2. Score & filter
    df_scored, daily_budget = score_and_filter(df, user_input)

    # 3. Geographic clustering
    n_days = user_input["duration_days"]
    df_scored, kmeans, scaler = cluster_by_geography(df_scored, n_days)

    # 4. Constraint optimization
    max_budget_per_day = daily_budget
    optimized = optimize_itinerary(
        df_scored, n_days, MAX_HOURS_PER_DAY, max_budget_per_day
    )

    # 5. Build JSON output
    itinerary_json = build_itinerary_json(optimized, user_input, n_days)

    output_filename = (
        f"itinerary_{user_input['governorate'].lower()}_{n_days}days.json"
    )
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(itinerary_json, f, ensure_ascii=False, indent=2)
    print(f"\n JSON saved: {output_filename}")
    print("\n JSON Preview (trip_summary):")
    print(json.dumps(itinerary_json["trip_summary"], ensure_ascii=False, indent=2))

    # 6. Validation
    validate_itinerary(itinerary_json, max_budget_per_day)

    # 7. Save artifacts
    if save:
        save_artifacts(kmeans, scaler, n_days, user_input,
                       MAX_HOURS_PER_DAY, max_budget_per_day)

    print("\nModel pipeline finished successfully")
    return itinerary_json

def load_data():
    return load_and_preprocess("places_all_v1.json")

def load_config():
    import json
    with open("pipeline_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def run_pipeline(
    governorate,
    budget_egp,
    duration_days,
    interests,
    df,
    config
):
    user_input = {
        "governorate": governorate,
        "budget_egp": budget_egp,
        "duration_days": duration_days,
        "interests": interests,
    }

    return pipeline_core(
    "places_all_v1.json",
    user_input,
    save=False
)
# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Travel Itinerary Recommendation Model")
    parser.add_argument("--data",        default="places_all_v1.json",
                        help="Path to places JSON file")
    parser.add_argument("--governorate", default="Luxor",
                        choices=["Cairo", "Giza", "Luxor", "Aswan"])
    parser.add_argument("--budget",      type=int, default=3000,
                        help="Total budget in EGP")
    parser.add_argument("--days",        type=int, default=3,
                        help="Trip duration in days")
    parser.add_argument("--interests",   nargs="+",
                        default=["historical", "pharaonic", "temple", "museum"],
                        help="List of interest keywords")
    parser.add_argument("--no-save",     action="store_true",
                        help="Skip saving model artifacts")

    args = parser.parse_args()

    USER_INPUT = {
        "governorate":   args.governorate,
        "budget_egp":    args.budget,
        "duration_days": args.days,
        "interests":     args.interests,
    }

    print("\n User Input:")
    for k, v in USER_INPUT.items():
        print(f"   {k}: {v}")
    print()

    pipeline_core(args.data, USER_INPUT, save=not args.no_save)
