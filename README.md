# Image Describe API (FastAPI)

Endpoint REST unico che riceve un'immagine (max 10MB), estrae dimensioni e 3 colori dominanti e genera una descrizione testuale usando:
- modello locale (Ollama) **oppure**
- servizio API remoto OpenAI-compatible (API key via env)

La modalità è configurabile via file `.env.*` selezionato con `ENV_FILE`.

## Requisiti
- Python 3.10+
- (opzionale) Ollama per modalità local

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt