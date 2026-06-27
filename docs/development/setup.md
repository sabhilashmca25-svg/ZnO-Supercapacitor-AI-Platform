# Local Development Setup

## Prerequisites

- Python 3.11+ (3.13 supported)
- Node.js 18+ LTS
- Git 2.38+
- **Git LFS** — required for the RF model (201 MB)

Install Git LFS once per machine:
```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# Windows — download installer from https://git-lfs.com
```

## Clone with LFS

```bash
git lfs install           # once per machine
git clone <repository-url>
cd ZnO_Supercapacitor_AI_Platform
```

## Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS / Linux

pip install -r requirements.txt

# Copy example config (optional — defaults work out of the box)
cp .env.example .env

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI:  http://localhost:8000/docs
- Health:      http://localhost:8000/api/v1/health

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

The Vite dev server proxies `/api/*` → `http://127.0.0.1:8000`.
No `.env` file needed for local development.

## One-Click Launcher

From the project root, run `START_APP.bat` (Windows) or `bash start_app.sh` (macOS/Linux).
The launcher handles venv creation, pip install, npm install, and browser opening automatically.

## Type Checking

```bash
cd frontend
npx tsc --noEmit      # zero errors expected
```

## Python Syntax Check

```bash
# From project root
python -m py_compile launcher.py stop_app.py
find backend/app -name "*.py" -exec python -m py_compile {} +
```
