import json
from pathlib import Path
# Save JSON in the same directory as the current script
AGENT_ID_PATH = Path(__file__).resolve().parent / "azure_assistant_agents_config.json"

def save_agent_id(name, agent_id):
    if AGENT_ID_PATH.exists():
        with open(AGENT_ID_PATH, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[name] = agent_id
    with open(AGENT_ID_PATH, "w") as f:
        json.dump(data, f)

def load_agent_id(name):
    if AGENT_ID_PATH.exists():
        with open(AGENT_ID_PATH, "r") as f:
            data = json.load(f)
        return data.get(name)
    return None
