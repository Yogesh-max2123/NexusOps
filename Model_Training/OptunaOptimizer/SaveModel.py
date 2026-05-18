import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from OptunaOptimizer.MLP import DynamicMLP

# MODULE: TEST SAVED MODEL / EXPORTED ARTIFACTS

def test_saved_model(model_path: str, best_params: dict, input_state: dict, X_test, y_test):
    """
    Loads the trained PyTorch model weights from disk and evaluates it on unseen test data.
    """
    print("\n=== STARTING FINAL MODEL TESTING ===")

    # 1. Recreate the blank model architecture
    input_dim = X_test.shape[1]
    output_dim = input_state["model_spec"]["output_dim"]

    # 2. Setup Device 
    # This automatically uses "cuda" (GPU) if available, otherwise falls back to "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    model = DynamicMLP(input_dim, output_dim, best_params)

   
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)

   
    model.eval()

    # 5. Prepare the test data tensor
    X_test_t = torch.FloatTensor(X_test).to(device)

    # 6. Run Inference
    print("Running predictions on the Test Set...")
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy().flatten()

    # 7. Calculate comprehensive metrics
    actuals = y_test.flatten()

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"R²   (Coefficient of Det.):     {r2:.4f}")

    
    print("\nSample Predictions:")
    for i in range(min(5, len(actuals))):
        print(f"  Actual: {actuals[i]:.2f} | Predicted: {predictions[i]:.2f} | Diff: {actuals[i] - predictions[i]:.2f}")

    return predictions