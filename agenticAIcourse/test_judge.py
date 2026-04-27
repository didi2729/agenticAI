##Veti construi un script Python standalone (judge.py) care:

##Citește un fișier JSON cu un scenariu de referință
##Creează o sesiune nouă în chatbot server printr-un request HTTP
##Parcurge tururile scenariului: pentru fiecare tur trimite mesajul utilizatorului și colectează răspunsul real al chatbot-ului
##Evaluează fiecare răspuns real față de răspunsul de referință folosind un apel separat la LLM (judge-ul)
##Afișează un raport cu scorurile per tur și un verdict final
##Șterge sesiunea la final


import json
import os
import requests
from uuid import uuid4
from dotenv import load_dotenv


load_dotenv()

# Configuration
CHATBOT_SERVER_URL = os.getenv("CHATBOT_SERVER_URL", "http://localhost:8000")
LLM_JUDGE_URL = os.getenv("LLM_JUDGE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "llama3")


def load_scenario(scenario_file: str) -> dict:
    """Load scenario from JSON file."""
    with open(scenario_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_session(server_url: str) -> str:
    """Create a new session in the chatbot server."""
    response = requests.post(f"{server_url}/api/sessions", timeout=30)
    response.raise_for_status()
    session_id = response.json().get("session_id")
    if not session_id:
        raise ValueError("No session_id in response")
    print(f"✓ Sesiune creată: {session_id}")
    return session_id


def send_message(server_url: str, session_id: str, message: str) -> str:
    """Send a message to the chatbot and get the response."""
    response = requests.post(
        f"{server_url}/api/sessions/{session_id}/chat",
        json={"message": message},
        timeout=60
    )
    response.raise_for_status()
    return response.json().get("response", "")


def evaluate_response(judge_url: str, judge_model: str, reference: str, actual: str) -> dict:
    """Evaluate actual response against reference using LLM judge."""
    prompt = f"""Evaluează răspunsul actual comparativ cu răspunsul de referință.

Răspuns de referință:
{reference}

Răspuns actual:
{actual}

Evaluare: Pe o scară de la 0 la 100, cât de bine se potrivește răspunsul actual cu cel de referință ca semantică și intenție?
Răspunde exclusiv în format JSON cu câmpurile:
- "score": (int 0-100) - scorul de similaritate
- "reason": (string) - justificarea scorului
"""
    payload = {
        "model": judge_model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(judge_url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json().get("response", "")
    
    # Parse JSON from response
    try:
        evaluation = json.loads(result)
    except json.JSONDecodeError:
        # Fallback: extract JSON from text
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            evaluation = json.loads(match.group())
        else:
            evaluation = {"score": 50, "reason": "Could not parse judge response"}
    
    return evaluation


def delete_session(server_url: str, session_id: str):
    """Delete the session."""
    response = requests.delete(f"{server_url}/api/sessions/{session_id}", timeout=30)
    response.raise_for_status()
    print(f"✓ Sesiune ștearsă: {session_id}")


def run_judge(scenario_file: str):
    """Main judge execution."""
    print(f"\n{'='*60}")
    print("JUDGE - Evaluare Răspunsuri Chatbot")
    print(f"{'='*60}\n")
    
    # 1. Load scenario
    print(f"📂 Se încarcă scenariul: {scenario_file}")
    scenario = load_scenario(scenario_file)
    print(f"   Scenariu: {scenario.get('name', 'Unknown')}")
    print(f"   Tururi: {len(scenario.get('turns', []))}\n")
    
    # 2. Create session
    session_id = create_session(CHATBOT_SERVER_URL)
    
    # 3. Process each turn
    results = []
    turns = scenario.get("turns", [])
    
    for i, turn in enumerate(turns, 1):
        print(f"\n--- Tur {i}/{len(turns)} ---")
        print(f"User: {turn['user_message']}")
        
        # Send message and get response
        actual_response = send_message(CHATBOT_SERVER_URL, session_id, turn['user_message'])
        print(f"Chatbot: {actual_response[:100]}...")
        
        # Evaluate against reference
        reference = turn.get("reference_response", "")
        evaluation = evaluate_response(LLM_JUDGE_URL, LLM_JUDGE_MODEL, reference, actual_response)
        
        print(f"   Scor: {evaluation.get('score', 'N/A')}/100")
        print(f"   Motivare: {evaluation.get('reason', 'N/A')}")
        
        results.append({
            "turn": i,
            "user_message": turn['user_message'],
            "reference": reference,
            "actual": actual_response,
            "score": evaluation.get('score', 0),
            "reason": evaluation.get('reason', '')
        })
    
    # 4. Delete session
    delete_session(CHATBOT_SERVER_URL, session_id)
    
    # 5. Generate report
    print(f"\n{'='*60}")
    print("RAPORT FINAL")
    print(f"{'='*60}")
    
    total_score = sum(r['score'] for r in results)
    avg_score = total_score / len(results) if results else 0
    
    print(f"\nScor mediu: {avg_score:.1f}/100")
    print(f"T verdict: {'PASS' if avg_score >= 70 else 'FAIL'}\n")
    
    for r in results:
        print(f"Tur {r['turn']}: {r['score']}/100")
    
    return {
        "scenario": scenario.get("name"),
        "total_turns": len(results),
        "average_score": avg_score,
        "verdict": "PASS" if avg_score >= 70 else "FAIL",
        "turns": results
    }


if __name__ == "__main__":
    import sys
    scenario_file = sys.argv[1] if len(sys.argv) > 1 else "scenario.json"
    result = run_judge(scenario_file)
    
    # Save report
    report_file = "judge_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Raport salvat: {report_file}")