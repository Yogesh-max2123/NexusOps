# ModelSmith 🚀

ModelSmith is an end-to-end scalable Machine Learning platform that bridges the gap between raw data and deployed deep learning models. It takes your dataset, target variable, and natural language requirements, and autonomously designs, trains, and validates a custom neural network architecture tailored to your specific use case.

## 🌟 Value Proposition (Why is it useful?)

Developing and tuning Neural Networks traditionally requires profound domain expertise in Machine Learning, hyperparameters, architecture design, and coding. ModelSmith automates this entire pipeline:
- **No-Code / Low-Code ML**: Users simply upload a CSV and provide natural language descriptions of their use case.
- **AI-Powered Architecture Design**: Uses Large Language Models (Google Gemini) to interpret user requirements and business context to synthesize constraints and search spaces for the model.
- **AutoML & Optimization**: Autonomously searches for the optimal hyperparameters and network topology using Optuna.
- **End-to-End MLOps**: Seamlessly manages the lifecycle from dataset ingestion (FastAPI + MongoDB) to asynchronous model training, and provides interactive endpoints to download artifacts or run immediate inferences right from the browser.

## 🏗️ Project Architecture

The repository is modularly structured into three core components:

1. **`app/` (Backend Ecosystem)**
   - A robust asynchronous **FastAPI** backend.
   - Manages user authentication (JWT + Bcrypt), session handling, and RESTful API endpoints.
   - Handles dataset uploads and uses **MongoDB (Motor)** to track user submissions and model artifacts.
2. **`frontend/` (User Interface)**
   - A highly interactive, premium-designed **Streamlit** dashboard.
   - Allows users to log in, define model requirements, kick-start background training tasks, and interactively test trained models using fresh test datasets.
3. **`Model_Training/` (Core AutoML Engine)**
   - The heart of the platform. Orchestrates a complete Machine Learning pipeline:
     - *InputStateBuilder* & *ConstraintEngine*: Interacts with the **Gemini API** to formalize training constraints.
     - *PrepareDataset*: Handles scaling and train/val/test splits via **Scikit-Learn**.
     - *OptunaOptimizer*: Intelligently explores hyperparameters to build PyTorch Multi-Layer Perceptrons (MLPs).
     - *Artifact Exporter*: Dumps out production-ready `.pth` model weights and `.json` deployment blueprints.

## 🛠️ Technology Stack & Requirements

### **Backend & APIs**
- **FastAPI**: High-performance async web framework.
- **Uvicorn**: ASGI server.
- **Pydantic**: Data validation and strict typing.
- **Celery / Redis / Kombu**: For task queues and asynchronous background processing (training jobs).
- **Passlib & Bcrypt**: Secure password hashing and cryptography.

### **Database & Storage**
- **MongoDB (PyMongo / Motor)**: Asynchronous NoSQL database for managing users and ML submission requests.

### **Machine Learning & AI**
- **PyTorch**: Core Deep Learning framework for constructing and training the neural networks.
- **Optuna**: State-of-the-art hyperparameter optimization framework.
- **Scikit-Learn**: For dataset preprocessing, evaluation metrics, and scaling.
- **Pandas & NumPy**: For efficient dataframe manipulations and mathematical operations.
- **Google Generative AI (Gemini)**: Serves as the constraint engine/LLM that extracts structural model requirements from user text.

### **Frontend UI**
- **Streamlit**: For rapid deployment of the application's data dashboards and interactive ML testing capabilities.
- **Requests**: For managing HTTP communication with the backend APIs.

## 🚀 Getting Started

### 1. Requirements & Environment
Ensure you have Python 3.9+ installed. Set up your global environment variables in a `.env` file (e.g., MongoDB credentials, GEMINI_API_KEY, JWT Secret).

### 2. Run the Backend API
Navigate to the root directory and start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run the Frontend Dashboard
In a new terminal window, navigate to the frontend directory:
```bash
streamlit run frontend/app.py
```

### 4. Direct CLI Model Training (Optional)
You can directly run the core ML engine without the API/UI by traversing to the `Model_Training` folder:
```bash
python main.py --csv_path "data.csv" --target "Price" --use_case "Predicting house prices." --req "Optimize for minimal MAE."
```
