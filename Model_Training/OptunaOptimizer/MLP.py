import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
import copy

# MODULE 3: OPTUNA HYPERPARAMETER TUNER & DYNAMIC MLP

class DynamicMLP(nn.Module):
    """
    A PyTorch Multi-Layer Perceptron (MLP) built dynamically.
    Each hidden layer has completely independent hyperparameters tuned by Optuna.
    """
    def __init__(self, input_dim, output_dim, params):
        super(DynamicMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # 1. Define the activation dictionary mapping
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=1),
            "sigmoid": nn.Sigmoid(),
            "linear": nn.Identity()
        }

        # 2. Build hidden layers dynamically and independently
        for i in range(params["num_hidden_layers"]):
            neurons = params[f"n_units_l{i}"]
            act_name = params[f"activation_l{i}"]
            drop_rate = params[f"dropout_l{i}"]
            use_bn = params[f"use_bn_l{i}"]

            
            layers.append(nn.Linear(current_dim, neurons))

           
            if use_bn:
                layers.append(nn.BatchNorm1d(neurons))

            
            chosen_activation = activation_map.get(act_name, nn.ReLU())
            layers.append(copy.deepcopy(chosen_activation))

            
            layers.append(nn.Dropout(drop_rate))

            
            current_dim = neurons

        
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

        
        self._initialize_weights(params["weight_init"])

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def suggest_hyperparameters(trial, search_space):
    """
    Translates the JSON search space constraints into Optuna trial suggestions,
    treating each hidden layer as an independent search space.
    """
    params = {}

    # 1. Suggest Global Hyperparameters
    global_keys = [
        "optimizer", "beta1", "learning_rate", "batch_size",
        "epochs", "early_stopping_patience", "grad_clip",
        "scheduler", "weight_init", "weight_decay"
    ]

    for key in global_keys:
        if key not in search_space: continue
        config = search_space[key]

        if config["type"] == "int":
            params[key] = trial.suggest_int(key, config["low"], config["high"], log=config.get("log", False))
        elif config["type"] == "float":
            params[key] = trial.suggest_float(key, config["low"], config["high"], log=config.get("log", False))
        elif config["type"] == "categorical":
            params[key] = trial.suggest_categorical(key, config["choices"])

    # 2. Determine the total number of hidden layers first
    n_layers_cfg = search_space["num_hidden_layers"]
    num_layers = trial.suggest_int("num_hidden_layers", n_layers_cfg["low"], n_layers_cfg["high"])
    params["num_hidden_layers"] = num_layers

    # 3. Suggest Layer-Specific Hyperparameters
    for i in range(num_layers):
        nc = search_space["num_neurons_base"]
        params[f"n_units_l{i}"] = trial.suggest_int(f"n_units_l{i}", nc["low"], nc["high"], log=nc.get("log", False))

        ac = search_space["activation_function"]
        params[f"activation_l{i}"] = trial.suggest_categorical(f"activation_l{i}", ac["choices"])

        dc = search_space["dropout_rate"]
        params[f"dropout_l{i}"] = trial.suggest_float(f"dropout_l{i}", dc["low"], dc["high"])

        bc = search_space["use_batch_norm"]
        params[f"use_bn_l{i}"] = trial.suggest_categorical(f"use_bn_l{i}", bc["choices"])

    return params


def create_objective(X_train, y_train, X_val, y_val, input_state, final_space):
    """
    Creates the objective function for Optuna to optimize.
    """
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    input_dim = X_train.shape[1]
    output_dim = input_state["model_spec"]["output_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    def objective(trial):
        params = suggest_hyperparameters(trial, final_space["search_space"])

        model = DynamicMLP(input_dim, output_dim, params).to(device)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,drop_last=True)

        if params["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"],
                                   weight_decay=params["weight_decay"], betas=(params["beta1"], 0.999))
        elif params["optimizer"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                    weight_decay=params["weight_decay"], betas=(params["beta1"], 0.999))
        else:
            optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"],
                                  weight_decay=params["weight_decay"])

        if params["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
        else:
            scheduler = None

        best_val_rmse = float('inf')
        patience_counter = 0

        for epoch in range(params["epochs"]):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
                optimizer.step()

            if scheduler:
                scheduler.step()

            model.eval()
            val_mse_sum = 0.0
            with torch.no_grad():
                
                for i in range(0, len(X_val_t), params["batch_size"]):
                    v_batch_X = X_val_t[i : i + params["batch_size"]].to(device)
                    v_batch_y = y_val_t[i : i + params["batch_size"]].cpu().numpy()
                    
                    v_preds = model(v_batch_X).cpu().numpy()
                    val_mse_sum += mean_squared_error(v_batch_y, v_preds) * len(v_batch_X)

            val_rmse = np.sqrt(val_mse_sum / len(X_val_t))

            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= params["early_stopping_patience"]:
                break

        return best_val_rmse

    return objective