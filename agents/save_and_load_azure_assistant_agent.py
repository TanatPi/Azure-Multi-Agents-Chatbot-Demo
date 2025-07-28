import json
from pathlib import Path

# === File paths for storing agent and file IDs ===
AGENT_ID_PATH = Path(__file__).resolve().parent / "azure_assistant_agents_config.json"
FILE_ID_PATH = Path(__file__).resolve().parent / "azure_assistant_file_ids.json"


# === Save & Load Azure Assistant Agent IDs ===
def save_agent_id(name: str, agent_id: str):
    """Save or update the ID of an assistant agent by name."""
    if AGENT_ID_PATH.exists():
        with open(AGENT_ID_PATH, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[name] = agent_id

    with open(AGENT_ID_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_agent_id(name: str) -> str | None:
    """Load a saved assistant agent ID by name."""
    if AGENT_ID_PATH.exists():
        with open(AGENT_ID_PATH, "r") as f:
            data = json.load(f)
        return data.get(name)
    return None


# === Save & Load File IDs uploaded to Azure Assistants ===
def save_file_id(filename: str, file_id: str):
    """Save or update the file ID associated with a given filename."""
    if FILE_ID_PATH.exists():
        with open(FILE_ID_PATH, "r", encoding="utf-8") as f:
            file_ids = json.load(f)
    else:
        file_ids = {}

    file_ids[filename] = file_id

    with open(FILE_ID_PATH, "w", encoding="utf-8") as f:
        json.dump(file_ids, f, indent=2)


def load_file_id(filename: str) -> str | None:
    """Load a saved file ID by filename."""
    if FILE_ID_PATH.exists():
        with open(FILE_ID_PATH, "r", encoding="utf-8") as f:
            file_ids = json.load(f)
        return file_ids.get(filename)
    return None
