import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
import copy
import logging
import os
from pymongo import MongoClient
from bson import ObjectId
from OptunaOptimizer.MLP import DynamicMLP
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("modelsmith")

def train_and_evaluate_final_model(best_params, X_train, y_train, X_test, y_test, input_state, submission_id=None, mlops_tracker=None):
    logger.info(" PHASE 5: FINAL MODEL TRAINING (LIVE DB SYNC)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, drop_last=True)

    input_dim = X_train.shape[1]
    output_dim = input_state["model_spec"]["output_dim"]
    model = DynamicMLP(input_dim, output_dim, best_params).to(device)
    criterion = nn.MSELoss()

    if best_params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
    elif best_params["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_params["epochs"]) if best_params["scheduler"] == "cosine" else None

    best_test_rmse = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    metrics_collected = {"epochs": [], "train_losses": [], "test_rmses": [],"test_maes": [], "test_r2s": []}

    
    collection = None
    if submission_id:
        try:
            mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            client = MongoClient(mongo_url)
            collection = client["modelsmith"]["submissions"]
            logger.info(" Connected to MongoDB for Live Updates")
        except Exception as e:
            logger.error(f" DB Connection failed: {e}")

    for epoch in range(best_params["epochs"]):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), best_params["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        if scheduler: scheduler.step()

        model.eval()
        t_preds_all = []
        t_actuals_all = []
        test_mse_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(X_test_t), best_params["batch_size"]):
                t_batch_X = X_test_t[i : i + best_params["batch_size"]].to(device)
                t_batch_y = y_test_t[i : i + best_params["batch_size"]].cpu().numpy()
                t_preds = model(t_batch_X).cpu().numpy()
                
                t_preds_all.extend(t_preds)
                t_actuals_all.extend(t_batch_y)
        test_rmse = np.sqrt(mean_squared_error(t_actuals_all, t_preds_all))
        test_mae = mean_absolute_error(t_actuals_all, t_preds_all)
        test_r2 = r2_score(t_actuals_all, t_preds_all)
        
        
        metrics_collected["epochs"].append(epoch + 1)
        metrics_collected["train_losses"].append(float(train_loss))
        metrics_collected["test_rmses"].append(float(test_rmse))
        metrics_collected["test_maes"].append(float(test_mae))  
        metrics_collected["test_r2s"].append(float(test_r2))    
        
        
        if collection is not None and submission_id:
            try:
                collection.update_one(
                    {"_id": ObjectId(submission_id)},
                    {"$set": {"model_config_json.metrics": metrics_collected}}
                )
            except Exception:
                pass
        
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t)
        final_rmse = float(np.sqrt(mean_squared_error(y_test_t.cpu().numpy(), final_preds.cpu().numpy())))
        
    
    return model, final_rmse, metrics_collected