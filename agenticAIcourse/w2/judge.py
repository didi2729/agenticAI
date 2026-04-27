import json
import os
import re
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHATBOT_SERVER_URL = os.getenv("CHATBOT_SERVER_URL", "http://localhost:8000")

judge_client = OpenAI(
    api_key=os.environ["JUDGE_API_KEY"],
    base_url=os.environ["JUDGE_BASE_URL"],
)
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

PASS_THRESHOLD = 3.5

EVAL_PROMPT = """\
Esti un evaluator strict. Scorifica raspunsul chatbot-ului pe trei dimensiuni.

Intrebarea utilizatorului:
{question}

Raspuns de referinta (calitate ridicata, acopera punctele esentiale):
{reference}

Raspunsul real al chatbot-ului:
{actual}

Scorifica fiecare dimensiune de la 1 la 5:
- accuracy  : 5 = acopera toate punctele din referinta fara erori; 1 = omite esentialul sau contine erori factuale
- clarity   : 5 = exceptional de clar si bine structurat; 1 = confuz sau greu de urmarit
- relevance : 5 = raspunde direct la intrebare, ramane pe subiect; 1 = off-topic sau ignora intrebarea

Ofera o singura propozitie de justificare.

Returneaza EXCLUSIV un obiect JSON valid, fara text suplimentar:
{{
  "accuracy": <int 1-5>,
  "clarity": <int 1-5>,
  "relevance": <int 1-5>,
  "justification": "<o propozitie>"
  "verdict": "PASS" sau "FAIL"
}}"""


def load_scenario(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_session(system_prompt: str | None) -> str:
    resp = requests.post(
        f"{CHATBOT_SERVER_URL}/sessions",
        json={"system_prompt": system_prompt},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["session_id"]


def send_message(session_id: str, content: str) -> dict:
    resp = requests.post(
        f"{CHATBOT_SERVER_URL}/sessions/{session_id}/messages",
        json={"content": content},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def delete_session(session_id: str) -> None:
    resp = requests.delete(
        f"{CHATBOT_SERVER_URL}/sessions/{session_id}",
        timeout=30,
    )
    resp.raise_for_status()


def evaluate_turn(question: str, reference: str, actual: str) -> dict:
    prompt = EVAL_PROMPT.format(
        question=question,
        reference=reference,
        actual=actual,
    )
    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    judge_prompt_tokens = response.usage.prompt_tokens
    judge_completion_tokens = response.usage.completion_tokens

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        scores = json.loads(match.group()) if match else {}

    for dim in ("accuracy", "clarity", "relevance"):
        val = scores.get(dim)
        if not isinstance(val, (int, float)) or not (1 <= val <= 5):
            scores[dim] = 1

    avg = (scores["accuracy"] + scores["clarity"] + scores["relevance"]) / 3
    scores["average"] = round(avg, 2)
    scores["verdict"] = "PASS" if avg >= PASS_THRESHOLD else "FAIL"
    scores["_judge_prompt_tokens"] = judge_prompt_tokens
    scores["_judge_completion_tokens"] = judge_completion_tokens

    return scores


def run_scenario(scenario_path: str) -> None:
    scenario = load_scenario(scenario_path)

    print(f"\n{'=' * 65}")
    print(f"Scenariu : {scenario['scenario_id']}")
    print(f"Descriere: {scenario['description']}")
    print(f"{'=' * 65}\n")

    session_id = create_session(scenario.get("system_prompt"))
    print(f"Sesiune creata: {session_id}\n")

    turns = scenario["turns"]

    total_chatbot_prompt = 0
    total_chatbot_completion = 0
    total_judge_prompt = 0
    total_judge_completion = 0
    passed = 0

    for i, turn in enumerate(turns, 1):
        print(f"--- Tur {i}/{len(turns)} ---")
        print(f"User: {turn['user']}")

        msg_resp = send_message(session_id, turn["user"])
        actual = msg_resp["content"]
        total_chatbot_prompt += msg_resp["prompt_tokens"]
        total_chatbot_completion += msg_resp["completion_tokens"]

        preview = actual[:150] + ("..." if len(actual) > 150 else "")
        print(f"Chatbot: {preview}")

        scores = evaluate_turn(turn["user"], turn["reference_output"], actual)
        total_judge_prompt += scores.pop("_judge_prompt_tokens")
        total_judge_completion += scores.pop("_judge_completion_tokens")

        if scores["verdict"] == "PASS":
            passed += 1

        print(
            f"  Acuratete: {scores['accuracy']}/5  "
            f"Claritate: {scores['clarity']}/5  "
            f"Relevanta: {scores['relevance']}/5"
        )
        print(f"  Medie: {scores['average']}  Verdict: {scores['verdict']}")
        print(f"  Justificare: {scores.get('justification', '-')}\n")

    delete_session(session_id)
    print(f"Sesiune stearsa: {session_id}\n")

    print(f"{'=' * 65}")
    print("RAPORT FINAL")
    print(f"{'=' * 65}")
    print(f"Tururi PASS: {passed}/{len(turns)}")
    print(f"Verdict global: {'PASS' if passed == len(turns) else 'FAIL'}")
    print(f"\nTokeni chatbot — input: {total_chatbot_prompt}  output: {total_chatbot_completion}")
    print(f"Tokeni judge   — input: {total_judge_prompt}  output: {total_judge_completion}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python judge.py scenarios/<scenariu>.json")
        sys.exit(1)
    run_scenario(sys.argv[1])
