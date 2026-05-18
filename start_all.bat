@echo off
echo ===================================================
echo Starting ModelSmith Microservices Pipeline...
echo ===================================================

:: 1. Start MLflow in a new terminal window
echo Starting MLflow Server...
start "MLflow UI" cmd /k ".\venv\Scripts\activate && mlflow ui --host 127.0.0.1 --port 5000"

echo Starting Main Backend API...
start "ModelSmith - Main API" cmd /k "call .\venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo Starting Celery Worker...
start "ModelSmith - Celery Worker" cmd /k "call .\venv\Scripts\activate && python -m celery -A app.celery_config:celery_app worker --loglevel=info --pool=solo"

echo Starting Data Cleaner API...
start "ModelSmith - Data Cleaner API" cmd /k "call .\venv\Scripts\activate && cd data-cleaner-api && uvicorn main:app --reload --host 0.0.0.0 --port 8001"

echo Starting Model Training API...
start "ModelSmith - Training API" cmd /k "call .\venv\Scripts\activate && cd Model_Training\OptunaOptimizer && python main.py"

echo Starting Model Training Worker...
start "ModelSmith - Training Worker" cmd /k "call .\venv\Scripts\activate && cd Model_Training\OptunaOptimizer && python worker.py"

echo Starting Streamlit Frontend...
:: Thoda sa delay de rahe hain taaki backend pehle start ho jaye
timeout /t 3 /nobreak > NUL
start "ModelSmith - Frontend UI" cmd /k "call .\venv\Scripts\activate && streamlit run frontend/app.py"

echo All services have been launched in separate windows!
pause