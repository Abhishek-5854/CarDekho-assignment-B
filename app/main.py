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
            # Clean and convert data
            row["seating_capacity"] = int(float(row["seating_capacity"]))
            row["ex_showroom_price"] = clean_price(row["ex_showroom_price"])
            row["on_road_price"] = clean_price(row["on_road_price"])
            row["mileage"] = clean_mileage(row["mileage"])
            row["engine"] = clean_engine(row["engine"])
            row["max_power"] = clean_power(row["max_power"])
            row["max_torque"] = clean_torque(row["max_torque"])

            # Handle missing or invalid safety ratings
            for key in ["NHTSA_Safety_Rating", "Global_NCAP_Safety_Rating"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(float(row[key]))
                    except (ValueError, TypeError):
                        row[key] = None
                else:
                    row[key] = None

            cars.append(Car(**row))
    return cars


def clean_price(price_str: str) -> str:
    # Remove non-numeric characters and "Rs."
    cleaned_price = re.sub(r"[^\\d.]", "", price_str).replace("Rs.", "").strip()
    return cleaned_price


def clean_mileage(mileage_str: str) -> str:
    # Extract numeric part and "kmpl"
    match = re.search(r"(\d+\\.?\\d*)\\s*kmpl", mileage_str)
    if match:
        return match.group(0)
    return mileage_str  # Return original if no match


def clean_engine(engine_str: str) -> str:
    # Extract numeric part and "cc"
    match = re.search(r"(\\d+)\\s*cc", engine_str)
    if match:
        return match.group(0)
    return engine_str  # Return original if no match


def clean_power(power_str: str) -> str:
    # Extract numeric part and "bhp"
    match = re.search(r"(\d+\\.?\\d*)\\s*bhp", power_str)
    if match:
        return match.group(0)
    return power_str  # Return original if no match


def clean_torque(torque_str: str) -> str:
    # Extract numeric part and "Nm"
    match = re.search(r"(\d+)\\s*Nm", torque_str)
    if match:
        return match.group(0)
    return torque_str  # Return original if no match


@app.get(\"/cars\", response_model=List[Car])
async def get_cars() -> List[Car]:
    return CAR_DATA


@app.post(\"/chat/questionnaire\")
async def chat_questionnaire(message: Message) -> Dict[str, Any]:
    response = get_questionnaire_preferences(message.content)
    return {\"response\": response}


@app.post(\"/chat/recommendation\")
async def chat_recommendation(message: Message) -> Dict[str, Any]:
    if not CAR_DATA:
        raise HTTPException(status_code=500, detail=\"Car data not loaded.\")

    # Combine all car details into a single string for the LLM
    car_details_str = \"\\n\".join([str(car.dict()) for car in CAR_DATA])
    combined_input = f\"User Message: {message.content}\\nAvailable Cars: {car_details_str}\"
    summary = generate_recommendation_summary(combined_input)
    return {\"summary\": summary}



