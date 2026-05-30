import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / ".env"

dotenv.load_dotenv(dotenv_path=DOTENV_PATH)

if not any(key in os.environ for key in ["GROQ_API_KEY", "groq_api_key"]):
    raise RuntimeError(
        "AI API key is required. Set GROQ_API_KEY or groq_api_key in .env or in the environment."
    )

from src.chat import (
    get_questionnaire_preferences,
    generate_recommendation_summary,
    is_questionnaire_agent_ready,
)

DATA_FILE = BASE_DIR / "src" / "car.csv"

app = FastAPI(
    title="CarDekho Recommendation API",
    description="A simple FastAPI backend that recommends cars based on user preferences and uses a questionnaire and recommendation agent flow.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAR_DATA: List[Dict[str, Any]] = []


class RecommendationRequest(BaseModel):
    user_message: str = Field(..., example="I want a family SUV under 18 lakhs with automatic transmission")
    top_k: int = Field(3, ge=1, le=5, description="Number of matching cars to return")


class CarMatch(BaseModel):
    car_name: str
    brand: str
    variant: str
    body_type: Optional[str] = None
    seating_capacity: Optional[str] = None
    fuel_type: Optional[str] = None
    transmission_type: Optional[str] = None
    drivetrain: Optional[str] = None
    ex_showroom_price_avg: Optional[float] = None
    on_road_price_avg: Optional[float] = None
    target_persona: Optional[str] = None
    best_use_case: Optional[str] = None
    worst_tradeoff: Optional[str] = None


class RecommendationResponse(BaseModel):
    query: str
    preferences: Optional[Dict[str, Any]] = None
    matches: List[CarMatch] = []
    assistant_response: str
    next_question: Optional[str] = None


def parse_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    cleaned = raw.replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_car_data() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find car data at {DATA_FILE}")

    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    numeric_columns = [
        "starting_price",
        "ending_price",
        "ex_showroom_price_avg",
        "on_road_price_avg",
        "safety_score",
        "economy_score",
        "family_comfort_score",
        "feature_score",
        "performance_score",
    ]

    for row in rows:
        for col in numeric_columns:
            if col in row:
                row[col] = parse_number(row[col])

    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_budget(query: str) -> Optional[float]:
    query = query.lower().replace(",", "")
    if "crore" in query:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(crore|cr)", query)
        if match:
            return float(match.group(1)) * 10_000_00
    if "lakh" in query or "lac" in query:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac)", query)
        if match:
            return float(match.group(1)) * 100_000
    if "thousand" in query or re.search(r"\d+\s*k\b", query):
        match = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|k)\b", query)
        if match:
            return float(match.group(1)) * 1_000
    try:
        return float(query)
    except ValueError:
        return None


def score_car_by_preferences(row: Dict[str, Any], preferences: Dict[str, Any]) -> float:
    score = 0.0
    if not preferences:
        return score

    for pref_key in ["body_type", "fuel_type", "transmission_type", "drivetrain", "brand", "target_persona"]:
        pref_value = normalize_text(preferences.get(pref_key, ""))
        row_value = normalize_text(row.get(pref_key, ""))
        if pref_value and pref_value in row_value:
            score += 4.0

    if preferences.get("budget"):
        budget_value = preferences.get("budget")
        if isinstance(budget_value, str):
            budget_amount = parse_budget(budget_value)
        else:
            try:
                budget_amount = float(budget_value)
            except Exception:
                budget_amount = None
        price = row.get("ex_showroom_price_avg") or row.get("on_road_price_avg") or 0
        if budget_amount and price and price <= budget_amount:
            score += 6.0

    if normalize_text(preferences.get("use_case", "")) and normalize_text(preferences.get("use_case", "")) in normalize_text(row.get("best_use_case", "")):
        score += 3.0

    if preferences.get("safety_priority") and normalize_text(str(preferences.get("safety_priority"))) in normalize_text(row.get("target_persona", "")):
        score += 2.0

    score += (row.get("feature_score") or 0) / 10.0
    score += (row.get("performance_score") or 0) / 15.0
    score += (row.get("economy_score") or 0) / 15.0
    score += (row.get("safety_score") or 0) / 20.0
    return score


def query_car_database(preferences: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for row in CAR_DATA:
        score = score_car_by_preferences(row, preferences)
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] == 0:
        scored = [(0.0, row) for row in sorted(CAR_DATA, key=lambda row: ((row.get("safety_score") or 0), (row.get("feature_score") or 0)), reverse=True)[:top_k]]

    selected = [row for _, row in scored[:top_k]]
    return [car_to_match_object(row) for row in selected]


def car_to_match_object(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "car_name": row.get("car_name", "Unknown"),
        "brand": row.get("brand", "Unknown"),
        "variant": row.get("variant", "Unknown"),
        "body_type": row.get("body_type"),
        "seating_capacity": row.get("seating_capacity"),
        "fuel_type": row.get("fuel_type"),
        "transmission_type": row.get("transmission_type"),
        "drivetrain": row.get("drivetrain"),
        "ex_showroom_price_avg": row.get("ex_showroom_price_avg"),
        "on_road_price_avg": row.get("on_road_price_avg"),
        "target_persona": row.get("target_persona"),
        "best_use_case": row.get("best_use_case"),
        "worst_tradeoff": row.get("worst_tradeoff"),
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "CarDekho recommendation API is running."}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    global CAR_DATA
    if not CAR_DATA:
        try:
            CAR_DATA = load_car_data()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    questionnaire_result = get_questionnaire_preferences(request.user_message)
    preferences = questionnaire_result.get("preferences") or {}
    next_question = questionnaire_result.get("next_question")

    if next_question and not preferences:
        return RecommendationResponse(
            query=request.user_message,
            preferences=preferences,
            matches=[],
            assistant_response="Please answer the following question so I can narrow down the best cars for you.",
            next_question=next_question,
        )

    matches = query_car_database(preferences, request.top_k)
    assistant_response = generate_recommendation_summary(matches, request.user_message)

    return RecommendationResponse(
        query=request.user_message,
        preferences=preferences,
        matches=matches,
        assistant_response=assistant_response,
        next_question=None,
    )



