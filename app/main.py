import csv
import os
import re
from contextlib import asynccontextmanager
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
        "AI API key is required. Set GROQ_API_KEY or groq_api_key in backend/.env or in the environment."
    )

from src.chat import (
    get_questionnaire_preferences,
    generate_recommendation_summary,
    is_questionnaire_agent_ready,
)

DATA_FILE = BASE_DIR / "src" / "car.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CAR_DATA
    CAR_DATA = load_car_data()
    if is_questionnaire_agent_ready():
        print("Questionnaire agent is initialized and ready.")
    yield

app = FastAPI(
    title="CarDekho Recommendation API",
    description="A simple FastAPI backend that recommends cars based on user preferences and uses a questionnaire and recommendation agent flow.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    content: str


class Car(BaseModel):
    body_type: str
    seating_capacity: int
    fuel_type: str
    transmission_type: str
    drivetrain: str
    ex_showroom_price: str
    on_road_price: str
    brand: str
    model: str
    variant: str
    features: str
    mileage: str
    engine: str
    max_power: str
    max_torque: str
    nhtsa_safety_rating: Optional[int] = Field(None, alias="NHTSA_Safety_Rating")
    global_ncpa_safety_rating: Optional[int] = Field(None, alias="Global_NCAP_Safety_Rating")


def load_car_data(file_path: Path = DATA_FILE) -> List[Car]:
    cars = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Map CSV columns to Car model fields with safe fallbacks
            seating = row.get("seating_capacity") or row.get("seating") or "0"
            try:
                seating = int(float(seating))
            except Exception:
                seating = 0

            ex_price_raw = row.get("ex_showroom_price") or row.get("ex_showroom_price_avg") or row.get("starting_price") or ""
            on_road_raw = row.get("on_road_price") or row.get("on_road_price_avg") or row.get("ending_price") or ""

            mileage_raw = row.get("mileage") or row.get("mileage_kmpl") or row.get("city_mileage_kmpl") or ""
            engine_raw = row.get("engine") or row.get("engine_displacement_cc") or ""
            max_power_raw = row.get("max_power") or row.get("max_power_bhp") or row.get("max_power_bhp") or ""
            max_torque_raw = row.get("max_torque") or row.get("max_torque_nm") or ""

            safe_row = {
                "body_type": row.get("body_type", ""),
                "seating_capacity": seating,
                "fuel_type": row.get("fuel_type", ""),
                "transmission_type": row.get("transmission_type", ""),
                "drivetrain": row.get("drivetrain", ""),
                "ex_showroom_price": clean_price(ex_price_raw) if ex_price_raw else "",
                "on_road_price": clean_price(on_road_raw) if on_road_raw else "",
                "brand": row.get("brand", ""),
                "model": row.get("model", ""),
                "variant": row.get("variant", ""),
                "features": row.get("features", ""),
                "mileage": clean_mileage(mileage_raw) if mileage_raw else "",
                "engine": (f"{engine_raw} cc" if engine_raw and engine_raw.isdigit() else engine_raw),
                "max_power": clean_power(max_power_raw) if max_power_raw else "",
                "max_torque": clean_torque(max_torque_raw) if max_torque_raw else "",
                "NHTSA_Safety_Rating": None,
                "Global_NCAP_Safety_Rating": None,
                "car_name": row.get("car_name", ""),
            }

            cars.append(Car(**safe_row))
    return cars


def clean_price(price_str: str) -> str:
    # Remove non-numeric characters and "Rs."
    cleaned_price = re.sub(r"[^\d.]", "", price_str).replace("Rs.", "").strip()
    return cleaned_price


def clean_mileage(mileage_str: str) -> str:
    # Extract numeric part and "kmpl"
    match = re.search(r"(\d+\.?\d*)\s*kmpl", mileage_str, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return mileage_str  # Return original if no match


def clean_engine(engine_str: str) -> str:
    # Extract numeric part and "cc"
    match = re.search(r"(\d+)\s*cc", engine_str, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return engine_str  # Return original if no match


def clean_power(power_str: str) -> str:
    # Extract numeric part and "bhp"
    match = re.search(r"(\d+\.?\d*)\s*bhp", power_str, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return power_str  # Return original if no match


def clean_torque(torque_str: str) -> str:
    # Extract numeric part and "Nm"
    match = re.search(r"(\d+)\s*Nm", torque_str, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return torque_str  # Return original if no match


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/recommend")
async def recommend(message: Message) -> Dict[str, Any]:
    if not CAR_DATA:
        raise HTTPException(status_code=500, detail="Car data not loaded.")

    summary = generate_recommendation_summary(CAR_DATA, message.content)
    return {"summary": summary}



