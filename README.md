# 🚀 NexusOps: Distributed & Fault-Tolerant Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-Distributed-37814A?logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Message_Broker-DC382D?logo=redis&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Optimization-254A81)

**NexusOps** is an enterprise-grade, horizontally scalable Machine Learning architecture designed to automate deep learning training and hyperparameter tuning. It is specifically engineered to run complex ML workloads (like Kaggle datasets) on highly constrained cloud environments (e.g., 512MB RAM free-tier instances) by separating computation from the API layer using a distributed worker-node cluster.

---

## 🌟 The Engineering Masterstroke
Training Deep Learning models on 512MB RAM servers usually results in instant Out-Of-Memory (OOM) crashes. NexusOps solves this through **Horizontal Scaling and Memory Orchestration**. 
By deploying multiple independent worker nodes connected via an **Upstash Redis Message Broker**, the system dynamically distributes hyperparameter tuning trials. If one node fails, the system recovers. If more power is needed, more free-tier nodes are attached. It brings data-center level distributed computing to zero-cost cloud hosting.

---

## 🏗️ Detailed Architecture & Workflow

NexusOps operates on a fully automated, 7-step sequential and distributed pipeline:

### 1. User Input & Input State Builder (Streamlit)
* The user interacts with a responsive Streamlit UI.
* Inputs include the raw CSV dataset, the target column to predict, and a natural language description of the "Use Case".
* The UI packages this into an immutable Input State and sends it to the FastAPI backend, tracking progress via a unique Submission ID.

### 2. LLM-Powered Constraint Engine (Gemini API)
* **The Problem:** Blindly building neural networks can explode memory.
* **The Solution:** The backend sends the dataset metadata and user "Use Case" to the **Google Gemini API**. The LLM acts as an architect, generating strict JSON constraints (e.g., max hidden layers, specific activation functions, dropout boundaries) tailored to the specific dataset size to prevent memory exhaustion during training.

### 3. Automated Dataset Cleaning & Preparation
* Workers download the dataset and perform automated preprocessing.
* **Memory Optimization:** Downcasts `float64` to `float32` to instantly halve the RAM footprint.
* Handles missing values, performs categorical encoding (One-Hot), and splits data into Train/Val/Test subsets. PyTorch DataLoaders are configured with optimized batch sizes to stream data into memory efficiently.

### 4. Distributed Training & Optuna Optimization (Celery + Redis)
* This is the core engine. FastAPI pushes the training task to the **Redis Task Queue**.
* Multiple **Celery Worker Web Services** actively listen to this queue. They pull the task and begin parallel executions.
* **Optuna** utilizes `JournalRedisStorage` to maintain a centralized state. Worker A and Worker B can simultaneously run deep learning trials (e.g., 10 trials each) without overriding each other, dramatically reducing total execution time.

### 5. Final Model Extraction & Synchronous Training
* Once all distributed trials conclude, the system fetches the absolute best hyperparameter dictionary from the Redis database.
* A final, comprehensive PyTorch neural network is built using these best parameters and trained on the full dataset.

### 6. Executive Insights & MongoDB Sync
* Post-training, feature importance (e.g., finding out that 'smoker' dictates 'insurance charges') is calculated.
* This raw mathematical data is sent back to the **Gemini API** via a *Safe Fallback REST Call* to generate a human-readable Executive Summary.
* The trained model weights (`.pth`), configuration (`.json`), and metrics are permanently saved. The **MongoDB Atlas** document is updated to a `COMPLETED` state.

### 7. Live Metrics & Inference Testing (UI)
* The Streamlit frontend constantly polls the MongoDB database. Upon completion, it dynamically unlocks the dashboard.
* Users can view live $R^2$ scores, RMSE, and the AI-generated insights.
* **Inference Mode:** Users can upload a test dataset (without the target column), and the UI will run live predictions using the newly distributed-trained model.

---

## 🛠️ Tech Stack & Infrastructure
* **Frontend:** Streamlit
* **REST API:** FastAPI, Uvicorn
* **Distributed Queue & Broker:** Celery, Upstash Redis
* **Database:** MongoDB Atlas (NoSQL for state tracking)
* **AI & Machine Learning:** PyTorch, Optuna, Scikit-Learn, Pandas
* **Generative AI:** Google Gemini (Constraint Engine & Insight Generation)
* **Cloud Infrastructure:** Render (Multiple Web Services for API and Worker Nodes)
* **Uptime Management:** Cron-job / UptimeRobot (Bypassing free-tier sleep restrictions via dummy HTTP servers).

---

## ⚙️ Handling Cloud Constraints (The "How-To")
This project employs several hardcore software engineering tactics:
1. **Thread Throttling:** `torch.set_num_threads(1)` ensures multi-core spikes don't trigger silent memory kills on fractional vCPUs.
2. **Aggressive Garbage Collection:** `gc.collect()` and explicit cache clearing after every Optuna trial.
3. **Decoupled Fallbacks:** If the Gemini API fails (e.g., 503 Server Error), the system uses a `try-except` boundary to inject default text, ensuring the 20-minute training pipeline doesn't crash at the finish line.

---

## 🚀 Installation & Local Setup

To run this distributed cluster on your local machine:

**1. Clone the repository**
```bash
git clone [https://github.com/Yogesh-max2123/NexusOps.git](https://github.com/Yogesh-max2123/NexusOps.git)
cd NexusOps
```
**2. Create a Virtual Environment & Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
**3. Environment Variables (.env)
Create a .env file in the root directory:**
```bash
MONGODB_URL=your_mongodb_atlas_connection_string
REDIS_URL=your_upstash_redis_url
GEMINI_API_KEY=your_google_gemini_api_key
CLOUDINARY_URL=your_cloudinary_url (if used for file hosting)
```
**4.Start the Cluster (Open 3 separate terminals)**


Terminal 1: Start the MLflow Tracking Server (The Control Room)
```bash
mlflow ui
# Accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000)
```

Terminal 2: Start the FastAPI Backend
```bash
uvicorn app.main:app --reload
```
Terminal 3: Start the Celery Worker (The ML Engine)
```bash
celery -A app.celery_config worker --loglevel=info --pool=solo
```
Terminal 4: Start the Streamlit UI
```bash
streamlit run ui/app.py
```
---
*Engineered out of pure curiosity to crack cloud constraints, master MLOps, and build some seriously cool stuff.*
