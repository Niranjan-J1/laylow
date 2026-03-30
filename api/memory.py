import json
import os
from pathlib import Path
from datetime import datetime

MEMORY_FILE = Path(__file__).parent.parent / "memory.json"

def _load() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except:
            pass
    return {"facts": [], "conversations": []}

def _save(data: dict):
    MEMORY_FILE.write_text(json.dumps(data, indent=2))

def add_fact(fact: str):
    data = _load()
    # Avoid duplicates
    if fact not in data["facts"]:
        data["facts"].append(fact)
        _save(data)

def get_facts() -> list[str]:
    return _load()["facts"]

def save_conversation(messages: list[dict], title: str = ""):
    data = _load()
    data["conversations"].append({
        "title":     title or messages[0]["content"][:40] if messages else "Untitled",
        "timestamp": datetime.now().isoformat(),
        "messages":  messages
    })
    # Keep last 50 conversations
    data["conversations"] = data["conversations"][-50:]
    _save(data)

def get_conversations() -> list[dict]:
    return _load()["conversations"]

def clear_facts():
    data = _load()
    data["facts"] = []
    _save(data)

def extract_facts_from_message(content: str) -> list[str]:
    """Extract simple facts from user messages to remember."""
    facts = []
    triggers = [
        "my name is", "i am", "i'm", "i work", "i live",
        "i like", "i love", "i hate", "i prefer", "remember that",
        "don't forget", "keep in mind", "i have", "my job"
    ]
    lower = content.lower()
    for trigger in triggers:
        if trigger in lower:
            # Extract the sentence containing the trigger
            sentences = content.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|")
            for sentence in sentences:
                if trigger in sentence.lower() and len(sentence.strip()) > 10:
                    facts.append(sentence.strip())
    return facts