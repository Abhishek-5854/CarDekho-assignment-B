import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from langchain_groq import ChatGroq


dotenv.load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


QUESTIONNAIRE_SYSTEM_PROMPT = (
    "You are a questionnaire agent that collects car preferences from the user. "
    "Your job is to gather 10 preference details before the recommendation step. "
    "Use the data points below to ask short, direct follow-up questions until you have enough information to fill all required fields. "
    "Required preference keys: body_type, seating_capacity, fuel_type, transmission_type, drivetrain, budget, preferred_brand, use_case, safety_priority, and any other supporting preference detail. "
    "Do not return recommendation text. If you do not yet have all preference values, ask a single clear follow-up question. "
    "Only return valid JSON when you have collected all requested preferences and can provide the complete object."
)

RECOMMENDATION_SYSTEM_PROMPT = (
    "You are a helpful car recommendation assistant. "
    "You will be given a list of cars. Generate a response explaining the cars, what is best about each one, why the user should buy them, and then ask the user to choose one of the candidate cars. "
    "Return the chosen car details in JSON format if the user explicitly asks for a choice, otherwise provide a concise recommendation summary."
)

questionnaire_memory: List[Dict[str, str]] = []
recommendation_memory: List[Dict[str, str]] = []

questionnaire_model: Optional[Any] = None
recommendation_model: Optional[Any] = None


def _clean_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().strip('"').strip("'")


def _get_groq_api_key() -> Optional[str]:
    return _clean_api_key(os.environ.get("GROQ_API_KEY")) or _clean_api_key(os.environ.get("groq_api_key"))


groq_api_key = _get_groq_api_key()
if groq_api_key:
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.7,
        )
        questionnaire_model = llm
        recommendation_model = llm
    except Exception as exc:
        print(f"Failed to initialize Groq LLM: {exc}")
        questionnaire_model = None
        recommendation_model = None
else:
    questionnaire_model = None
    recommendation_model = None


def _build_messages(system_prompt: str, human_message: str, memory: List[Dict[str, str]]) -> List[tuple]:
    messages: List[tuple] = [("system", system_prompt)]
    for item in memory:
        messages.append((item["role"], item["content"]))
    messages.append(("human", human_message))
    return messages


def _append_memory(memory: List[Dict[str, str]], role: str, content: str) -> None:
    memory.append({"role": role, "content": content})


def _invoke_chat_model(model: Optional[Any], messages: List[tuple]) -> str:
    if model is None:
        return "\n".join(content for role, content in messages if role == "human")

    try:
        output = model.invoke(messages)
    except Exception as exc:
        return f"[LLM error: {exc}]"

    if isinstance(output, str):
        return output
    if hasattr(output, "content"):
        return getattr(output, "content")
    if hasattr(output, "text"):
        return getattr(output, "text")
    if isinstance(output, dict) and "content" in output:
        return output["content"]
    return str(output)


def is_questionnaire_agent_ready() -> bool:
    return questionnaire_model is not None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        snippet = text[first:last + 1]
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def get_questionnaire_preferences(user_message: str) -> Dict[str, Any]:
    prompt = user_message.strip()
    required_keys = [
        "body_type",
        "seating_capacity",
        "fuel_type",
        "transmission_type",
        "drivetrain",
        "budget",
        "preferred_brand",
        "use_case",
        "safety_priority",
    ]

    if questionnaire_model is None:
        return {
            "preferences": {"raw_input": prompt},
            "next_question": (
                "Please tell me your budget, body type, fuel type, or seating requirements so I can narrow down recommendations."
            ),
        }

    messages = _build_messages(QUESTIONNAIRE_SYSTEM_PROMPT, prompt, questionnaire_memory)
    output = _invoke_chat_model(questionnaire_model, messages)
    _append_memory(questionnaire_memory, "human", prompt)
    _append_memory(questionnaire_memory, "assistant", output)

    if output.startswith("[LLM"):
        return {"preferences": {}, "next_question": "I am having trouble reaching the recommendation engine. Please try again later."}

    preferences = _extract_json(output)
    if preferences:
        missing_keys = [
            key for key in required_keys if not preferences.get(key) or str(preferences.get(key)).strip().lower() in {"", "unknown", "n/a", "none"}
        ]
        if missing_keys:
            next_question = (
                "Please provide more details for the following preferences: "
                + ", ".join(missing_keys)
                + "."
            )
            return {"preferences": preferences, "next_question": next_question}
        return {"preferences": preferences, "next_question": None}

    return {"preferences": {}, "next_question": output.strip()}


def generate_recommendation_summary(cars: List[Dict[str, Any]], user_message: str) -> str:
    if not cars:
        return "I could not find a match for your preferences. Please try again with a few more details."

    car_lines = [
        f"- {car['car_name']} ({car['brand']} {car['variant']}): {car.get('body_type', 'N/A')}, {car.get('fuel_type', 'N/A')} priced around {int(car['ex_showroom_price_avg']) if car['ex_showroom_price_avg'] else 'N/A'}"
        for car in cars
    ]

    candidate_text = "\n".join(car_lines)
    user_prompt = (
        f"The user asked: '{user_message}'.\n\n"
        f"Review the candidate cars below and explain why each one would be a good fit. Keep the response concise and easy to read.\n\n"
        f"Candidate cars:\n{candidate_text}"
    )

    if recommendation_model is None:
        fallback = [
            f"{car['car_name']} ({car['brand']} {car['variant']}) - {car.get('body_type', 'Unknown body type')}, {car.get('fuel_type', 'Unknown fuel')} priced around {int(car['ex_showroom_price_avg']) if car['ex_showroom_price_avg'] else 'N/A'}."
            for car in cars
        ]
        return "LLM recommendation agent is not configured. Here are the top matches:\n" + "\n".join(fallback)

    messages = _build_messages(RECOMMENDATION_SYSTEM_PROMPT, user_prompt, recommendation_memory)
    output = _invoke_chat_model(recommendation_model, messages)
    if output.startswith("[LLM"):
        fallback = [
            f"{car['car_name']} ({car['brand']} {car['variant']}) - {car.get('body_type', 'Unknown body type')}, {car.get('fuel_type', 'Unknown fuel')} priced around {int(car['ex_showroom_price_avg']) if car['ex_showroom_price_avg'] else 'N/A'}."
            for car in cars
        ]
        return "LLM recommendation agent is unavailable due to quota or API error. Here are the top matches:\n" + "\n".join(fallback)

    _append_memory(recommendation_memory, "human", user_prompt)
    _append_memory(recommendation_memory, "assistant", output)
    return output

