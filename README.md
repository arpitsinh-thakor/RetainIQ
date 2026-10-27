# RetainIQ — Next.js + FastAPI

A portfolio demo for customer churn decision intelligence. The frontend is Next.js 14, React, TypeScript, Tailwind CSS, Recharts and Lucide icons. The backend remains Python/FastAPI and reuses the existing RetainIQ ML modules.

## Requirements
- Python 3.10+ (3.11/3.12 recommended)
- Node.js 18.17+ (Node 20 LTS recommended)

## 1. Python backend (terminal 1)
Run commands from the project root (`RetainIQ_NextJS`, where `api.py` is located):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_pipeline
python -m src.modeling
uvicorn api:app --reload
```

API docs: http://127.0.0.1:8000/docs

## 2. Next.js frontend (terminal 2)
From the project root:

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. If `.env.local` already exists, do not overwrite it; make sure it contains `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`.

## Pages
- `/` Overview dashboard (live dataset metrics, contract breakdown, customer watchlist)
- `/customers` Customer explorer with search and heuristic risk-segment filter
- `/predict` Form calling `POST /predict`
- `/analyst` Chat UI calling `POST /ask`

## API endpoints
- `GET /health`
- `GET /metrics`
- `GET /customers?limit=100&search=&risk=all`
- `POST /predict`
- `POST /explain`
- `POST /ask`

## Important notes
- The data pipeline generates synthetic demo data; it is not real customer data.
- The churn pulse chart uses illustrative historical points except for the latest point, which uses the current generated dataset rate. It is labeled accordingly.
- The customer risk segment is a transparent heuristic and is not the trained model's probability.
- Model inference and chat depend on the included Python modules and installed dependencies. The LangChain LLM path requires a configured API key if that path is enabled in `src/analyst.py`.
- The backend allows CORS only from localhost/127.0.0.1 port 3000 for local development. Update the allowlist before deployment.

## Troubleshooting
- If `/metrics` says dataset not found, run `python -m src.data_pipeline` from the project root.
- If the browser cannot connect, confirm FastAPI is running on port 8000 and `.env.local` has the correct API URL; restart `npm run dev` after editing environment variables.
- If model files are missing, run `python -m src.modeling` after generating data.
