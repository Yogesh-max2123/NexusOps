import streamlit as st
import requests
import pandas as pd
import io
import plotly.graph_objects as go  # Added missing import for graphs to prevent crash
import time
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Dataset Hub", page_icon="🚀", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 20px;
        font-size: 3rem !important;
        font-weight: 800;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

if "access_token" not in st.session_state:
    st.session_state.access_token = None

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}"}

# ----------------- SIDEBAR AUTH -----------------
with st.sidebar:
    st.title("🛡️ Authentication")
    
    if st.session_state.access_token:
        st.success("✅ Logged in")
        if st.button("Logout"):
            st.session_state.access_token = None
            st.rerun()
    else:
        auth_mode = st.radio("Choose Action", ["Login", "Register"], horizontal=True)
        
        if auth_mode == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                res = requests.post(
                    f"{API_URL}/auth/login",
                    data={"username": username, "password": password}
                )
                if res.status_code == 200:
                    st.session_state.access_token = res.json()["access_token"]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        
        elif auth_mode == "Register":
            username = st.text_input("New Username")
            email = st.text_input("Email")
            password = st.text_input("New Password", type="password")
            if st.button("Register"):
                res = requests.post(
                    f"{API_URL}/auth/register",
                    json={"username": username, "email": email, "password": password}
                )
                if res.status_code == 201:
                    st.success("Registered successfully! Please login.")
                else:
                    try:
                        error_msg = res.json().get("detail", "Registration failed.")
                    except ValueError:
                        error_msg = f"Registration failed. Server returned {res.status_code}"
                    st.error(error_msg)

# ----------------- MAIN APP -----------------
st.markdown("<h1 class='main-header'>Neural Network Project Request Hub</h1>", unsafe_allow_html=True)

if not st.session_state.access_token:
    st.info("### 👋 Welcome! \nPlease log in using the sidebar on the left to submit your dataset and requirements.")
else:
    tab1, tab2, tab3 = st.tabs(["🚀 Submit New Request", "📂 My Submissions", "📊 Live Metrics"])
    
    with tab1:
        st.markdown("### 📤 Upload Your Dataset & Requirements")
        st.write("Fill out the details below to define the architecture and expectations you have from the model.")
        
        with st.container():
            with st.form("submission_form", border=True):
                dataset_file = st.file_uploader("Upload Dataset (CSV limits apply)", type=["csv", "txt"])
                target_column = st.text_input("🎯 Target Column Name", placeholder="e.g. Sales, Price, Quality")
                use_case = st.text_area("💼 Use Case Description", placeholder="Describe the business scenario where this model will be used.")
                requirement = st.text_area("📋 Specific Requirements", placeholder="List any particular nuances needed in the model output or architecture.")
                
                submitted = st.form_submit_button("Submit Request 🚀")
                
                if submitted:
                    if not dataset_file or not target_column or not use_case or not requirement:
                        st.error("⚠️ Please fill out all fields and upload a dataset.")
                    else:
                        with st.spinner("⏳ Uploading directly to Cloudinary and securing metadata... (this may take a few moments)"):
                            # Prepare the multipart payload
                            files = {"dataset": (dataset_file.name, dataset_file, dataset_file.type)}
                            data = {
                                "target_column": target_column,
                                "use_case": use_case,
                                "requirement": requirement
                            }
                            
                            try:
                                # Send to FastAPI
                                res = requests.post(f"{API_URL}/submit/", headers=get_headers(), files=files, data=data)
                                if res.status_code == 200:
                                    st.success("🎉 Successfully submitted request! Your dataset is backed up securely to Cloudinary.")
                                    st.json(res.json())
                                    st.balloons()
                                else:
                                    st.error(f"❌ Error submitting request: {res.text}")
                            except requests.exceptions.ConnectionError:
                                st.error("🔌 Could not connect to the Backend API. Ensure it is running on port 8000.")

    with tab2:
        st.markdown("### 📊 My Past Submissions")
        btn_col, _ = st.columns([1, 4])
        if btn_col.button("🔄 Refresh"):
            st.rerun()

        try:
            res = requests.get(f"{API_URL}/submit/", headers=get_headers())
            if res.status_code == 200:
                submissions = res.json()
                if not submissions:
                    st.info("You haven't submitted any datasets yet.")
                else:
                    for sub in sorted(submissions, key=lambda x: x.get('created_at', ''), reverse=True):
                        # Ensure we always stringify _id reliably for keys
                        sub_id = str(sub.get('_id', sub.get('id', ''))) 
                        status = sub.get('status', 'pending')
                        
                        with st.expander(f"Dataset: {sub.get('target_column', 'Unknown')} | Status: {status.upper()}"):
                            # ✅ Delete Button placed at the top of the expander
                            col_del1, col_del2 = st.columns([4, 1])
                            with col_del1:
                                st.write(f"**Use Case:** {sub.get('use_case', 'N/A')}")
                                st.write(f"**Requirement:** {sub.get('requirement', 'N/A')}")
                                st.write(f"**Dataset URL:** [Download Data]({sub.get('dataset_url', '#')})")
                                st.write(f"**Submitted At:** {sub.get('created_at', 'N/A')}")
                            with col_del2:
                                import time
                                if st.button("🗑️ Delete", key=f"del_{sub_id}", type="primary", use_container_width=True):
                                    with st.spinner("Deleting..."):
                                        del_res = requests.delete(f"{API_URL}/submit/{sub_id}", headers=get_headers())
                                        if del_res.status_code == 200:
                                            st.success("Deleted!")
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {del_res.text}")
                            
                            st.divider() # Visual separator
                            
                            if status in ['pending', 'failed']:
                                if st.button("🚀 Train Model", key=f"train_{sub_id}"):
                                    t_res = requests.post(f"{API_URL}/submit/{sub_id}/train", headers=get_headers())
                                    
                                    if t_res.status_code == 200:
                                        st.toast("🚀 Training Request Sent!")
                                        
                                        # 1. UI Elements create karna (Progress bar aur Text box)
                                        progress_bar = st.progress(5, text="Starting MLOps Pipeline...")
                                        status_message = st.empty()

                                        # 2. Polling Loop (Har 3 second me status check karega)
                                        import time
                                        is_training = True
                                        
                                        while is_training:
                                            try:
                                                # Backend se live status aur metrics mango
                                                check_res = requests.get(f"{API_URL}/submit/{sub_id}/live-metrics", headers=get_headers())

                                                if check_res.status_code == 200:
                                                    data = check_res.json()
                                                    current_status = data.get("status", "pending")
                                                    epochs_done = len(data.get("epochs", []))

                                                    # PHASE 1: Celery Queue
                                                    if current_status == "pending":
                                                        progress_bar.progress(15, text="⏳ Step 1/3: Queued in Server...")
                                                        status_message.info("Waiting for Celery worker to pick up the task...")

                                                    # PHASE 2 & 3: Training in Progress
                                                    elif current_status in ["in_progress", "training"]:
                                                        if epochs_done == 0:
                                                            # Optuna Phase (Metrics abhi aana shuru nahi hue)
                                                            progress_bar.progress(40, text="⚙️ Step 2/3: Running Optuna Hyperparameter Trials...")
                                                            status_message.warning("Testing multiple AI architectures to find the best one. (This takes ~1-2 mins)...")
                                                        else:
                                                            # Final Training Phase (Metrics aana shuru ho gaye)
                                                            progress_bar.progress(75, text=f"🧠 Step 3/3: Final Training (Epoch {epochs_done} done)...")
                                                            status_message.info("Best architecture found! Training final production model...")

                                                    # PHASE 4: Success
                                                    elif current_status == "completed":
                                                        progress_bar.progress(100, text="✅ Training Completely Finished!")
                                                        status_message.success("🎉 Model trained successfully! Switch to the 'Training Metrics' tab (Tab 3) to view AI Insights and Graphs.")
                                                        st.balloons()  # Celebration animation!
                                                        is_training = False
                                                        time.sleep(3) # 3 second wait balloon animation ke liye
                                                        st.rerun() # UI ko automatically refresh karne ke liye

                                                    # PHASE 5: Error handling
                                                    elif current_status in ["failed", "error"]:
                                                        progress_bar.empty()
                                                        status_message.error("❌ Training failed! Please check backend terminal logs.")
                                                        is_training = False

                                            except Exception as e:
                                                pass 

                                            if is_training:
                                                time.sleep(3)
                                                
                                    else:
                                        st.error(f"Error starting training: {t_res.text}")
                            
                            elif status == 'training':
                                st.info("⏳ Model is currently training... MLflow metrics will be available once training completes.")
                                    
                            elif status == 'completed':
                                st.success("✅ Training Completed - Model is ready!")
                                
                                # ✅ PERFECT 3-COLUMN LAYOUT
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # 🟢 1. DASHBOARD FIX: Simple message directing user to Tab 3
                                    if st.button("📊 View Live AI Dashboard", key=f"mlflow_{sub_id}", use_container_width=True):
                                        st.success("👆 Model selected! Please click on 'Tab 3' above to view insights.")
                                        
                                with col2:
                                    # 🟢 2. SMART DOWNLOAD FIX: JSON Config aur PTH Weights dono!
                                    if st.session_state.get(f"json_ready_{sub_id}"):
                                        
                                        # --- Button 1: Download JSON (Architecture & Metrics) ---
                                        config_data = st.session_state[f"json_data_{sub_id}"]
                                        import json
                                        json_string = json.dumps(config_data, indent=4)
                                        target_name = sub.get('target_column', 'model').replace(' ', '_').lower()
                                        
                                        st.download_button(
                                            label="📄 Download Config (.json)",
                                            data=json_string,
                                            file_name=f"{target_name}_config.json",
                                            mime='application/json',
                                            use_container_width=True,
                                            key=f"dl_json_ready_{sub_id}"
                                        )
                                        
                                        # --- Button 2: Download PTH (Weights & Biases) ---
                                        # Ye directly backend API se binary download karega (Streamlit ko hang nahi karega)
                                        pth_url = f"{API_URL}/submit/{sub_id}/model-file"
                                        st.markdown(f'''
                                            <a href="{pth_url}" download style="text-decoration: none;">
                                                <button style="width:100%; border:1px solid #1f77b4; color:white; background-color:transparent; padding:0.4rem; border-radius:4px; cursor:pointer; margin-top:2px;">
                                                    💾 Download Weights (.pth)
                                                </button>
                                            </a>
                                        ''', unsafe_allow_html=True)
                                        
                                    else:
                                        # Pehle DB se data load karne ka button
                                        if st.button("📥 Load Data for Export", key=f"fetch_dl_{sub_id}", use_container_width=True):
                                            with st.spinner("Fetching from Database..."):
                                                try:
                                                    single_res = requests.get(f"{API_URL}/submit/{sub_id}", headers=get_headers())
                                                    if single_res.status_code == 200:
                                                        single_data = single_res.json()
                                                        if 'model_config_json' in single_data and single_data['model_config_json']:
                                                            st.session_state[f"json_ready_{sub_id}"] = True
                                                            st.session_state[f"json_data_{sub_id}"] = single_data['model_config_json']
                                                            st.rerun()
                                                        else:
                                                            st.error("⚠️ Config Data not found in DB.")
                                                    else:
                                                        st.error(f"⚠️ Backend error: {single_res.text}")
                                                except Exception as e:
                                                    st.error(f"Error fetching data: {e}")
                                        
                                
                                with col3:
                                    if st.button("🧪 Test Model", key=f"test_btn_{sub_id}", use_container_width=True):
                                        st.session_state[f"show_test_{sub_id}"] = True
                                
                                # Test model section
                                if st.session_state.get(f"show_test_{sub_id}"):
                                    st.divider()
                                    st.markdown("#### 🧪 Test This Model")
                                    st.write("Upload a sample CSV file with the same features to get predictions.")
                                    test_file = st.file_uploader("Upload Test CSV", type=["csv"], key=f"test_{sub_id}")
                                    
                                    if test_file is not None:
                                        if st.button("🔮 Generate Predictions", key=f"pred_btn_{sub_id}"):
                                            with st.spinner("Wait for it... Running Inference..."):
                                                files = {"test_data": (test_file.name, test_file, test_file.type)}
                                                p_res = requests.post(f"{API_URL}/submit/{sub_id}/predict", headers=get_headers(), files=files)
                                                
                                                if p_res.status_code == 200:
                                                    st.success("Predictions generated successfully!")
                                                    
                                                    res_data = p_res.json()
                                                    csv_content = res_data["csv_data"]
                                                    metrics = res_data.get("metrics")
                                                    
                                                    if metrics:
                                                        st.markdown("### 📊 Test Metrics")
                                                        cols = st.columns(len(metrics))
                                                        for idx, (k, v) in enumerate(metrics.items()):
                                                            cols[idx].metric(label=k, value=f"{v:.4f}")
                                                    
                                                    # Convert CSV string back to dataframe to show nicely in UI
                                                    import pandas as pd
                                                    import io
                                                    df_preds = pd.read_csv(io.StringIO(csv_content))
                                                    st.dataframe(df_preds, use_container_width=True)
                                                    
                                                    st.download_button(
                                                        label="⬇️ Download Predictions CSV",
                                                        data=csv_content.encode("utf-8"),
                                                        file_name=f"predictions_{test_file.name}",
                                                        mime="text/csv",
                                                        key=f"dl_pred_{sub_id}"
                                                    )
                                                else:
                                                    st.error(f"Error making predictions: {p_res.text}")
            else:
                st.error("Failed to load submissions.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Backend API not running on port 8000.")
    with tab3:
        # Streamlit App (Tab 3 Code block replace)

        st.markdown("### 📊 Training Metrics Analysis")
        st.write("View detailed metrics and strategic insights from completed training.")
        
        try:
            # 1. Imports needed for updated UI graphs (Safe Imports inside try block)
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px # 🟢 NEW: for Bar Chart
            import pandas as pd
            import time

            # Get all submissions (Auth Required)
            res = requests.get(f"{API_URL}/submit/", headers=get_headers())
            
            if res.status_code == 200:
                submissions = res.json()
                completed = [s for s in submissions if s.get("status") == "completed"]
                
                if not completed:
                    st.info("📭 No completed trainings found yet.")
                else:
                    # Select model to view
                    options = [f"{s['target_column']} - {s.get('created_at', 'Date N/A')[:10]}" for s in completed]
                    selected = st.selectbox("Select Model to View Performance", options=options, key="metrics_select")
                    
                    if selected:
                        idx = options.index(selected)
                        sub = completed[idx]
                        sub_id = str(sub.get("_id", ""))
                        
                        st.markdown(f"#### 🎯 Forecast Model for: `{sub['target_column']}`")
                        
                        # Status cards
                        status_col, target_col, mlflow_col = st.columns(3)
                        status_col.metric("Workflow Status", "✅ Completed")
                        target_col.metric("Target Column", sub['target_column'])
                        mlflow_col.metric("MLflow Run ID", sub.get("mlflow_run_id", "N/A"))
                        
                        st.markdown("---")
                        
                        # Fetching Metrics spinner
                        st.markdown("#### 🚀 Engine Metrics Fetching...")
                        
                        with st.spinner("Fetching latest model engine parameters..."):
                            try:
                                metrics_res = requests.get(
                                    f"{API_URL}/submit/{sub_id}/live-metrics",
                                    headers=get_headers(),
                                    timeout=10
                                )
                                
                                if metrics_res.status_code == 200:
                                    data = metrics_res.json()
                                    
                                    # Extract Core Metrics arrays
                                    epochs = data.get("epochs", [])
                                    train_losses = data.get("train_losses", [])
                                    test_rmses = data.get("test_rmses", [])
                                    test_maes = data.get("test_maes", []) 
                                    test_r2s = data.get("test_r2s", [])   
                                    final_score = data.get("final_score")
                                    
                                    # Extract NEW Config Fields (Stats, Params, Insights)
                                    exec_stats = data.get("execution_stats", {})
                                    best_params = data.get("best_params", {})
                                    
                                    # 🟢 1. NAYA: Extract Feature Importance & LLM Summary
                                    f_importance_dict = data.get("feature_importance", {})
                                    llm_summary_paragraph = data.get("llm_executive_summary", "")

                                    st.success(f"✅ Loaded {len(epochs)} epochs of metrics.")
                                    
                                    if epochs and train_losses and test_rmses:
                                        
                                        # (A) BUSINESS FRIENDLY METRICS (MAE, R2 etc.)
                                        st.markdown("### 📊 Business Performance Metrics")
                                        ep_col, rmse_col, mae_col, r2_col = st.columns(4)
                                        ep_col.metric("Total Epochs", len(epochs))
                                        rmse_col.metric("Final RMSE (Error)", f"{test_rmses[-1]:.4f}" if test_rmses else "N/A")
                                        
                                        latest_mae = test_maes[-1] if test_maes else "N/A"
                                        latest_r2 = test_r2s[-1] if test_r2s else "N/A"
                                        
                                        if latest_mae != "N/A":
                                            mae_col.metric("MAE (Avg Error)", f"{latest_mae:.2f}")
                                        if latest_r2 != "N/A":
                                            r2_pct = max(0, latest_r2 * 100) 
                                            r2_col.metric("R² Score (Accuracy)", f"{r2_pct:.1f}%")

                                        st.markdown("---")

                                        # (B) NAYA: AI Insights Section (Put it prominent)
                                        st.markdown("### 🤖 Model Executive Summary & AI Insights")
                                        if llm_summary_paragraph:
                                            # Using st.info for nice formatting, but could use st.markdown/st.write
                                            st.info(f"👉 {llm_summary_paragraph}")
                                        else:
                                            st.info("Insights for this run have not been generated yet. (They are only available on newly trained models).")
                                            
                                        st.markdown("---")

                                        # (C) ENGINE DETAILS & HYPERPARAMETERS
                                        st.markdown("### ⚙️ Engine Details & Hyperparameters")
                                        exec_col, param_col = st.columns([1, 2])
                                        
                                        with exec_col:
                                            st.caption("⏱️ **Execution Stats**")
                                            if exec_stats:
                                                total_sec = exec_stats.get('total_time_seconds', 0)
                                                mins = int(total_sec // 60)
                                                secs = int(total_sec % 60)
                                                st.write(f"**Final Train Time:** {mins}m {secs}s")
                                                st.write(f"**Time / Epoch:** {exec_stats.get('time_per_epoch_seconds', 0):.2f}s")
                                                st.write(f"**Hardware:** 💻 {exec_stats.get('hardware_used', 'N/A')}")
                                            else:
                                                st.write("Execution stats N/A.")
                                                
                                        with param_col:
                                            st.caption("🧠 **Winning Hyperparameters**")
                                            if best_params:
                                                p_col1, p_col2, p_col3 = st.columns(3)
                                                with p_col1:
                                                    st.write(f"**Optimizer:** {best_params.get('optimizer', 'N/A').upper()}")
                                                    st.write(f"**learning Rate:** {best_params.get('learning_rate', 0):.4f}")
                                                with p_col2:
                                                    st.write(f"**Batch Size:** {best_params.get('batch_size', 'N/A')}")
                                                    st.write(f"**Scheduler:** {best_params.get('scheduler', 'N/A').title()}")
                                                with p_col3:
                                                    st.write(f"**Layers Config:** {best_params.get('num_hidden_layers', 'N/A')}")
                                                    
                                            else:
                                                st.write("Best parameters N/A.")

                                        st.markdown("---")

                                        # (D) NAYA: Feature Importance Dashboard Section
                                        st.markdown("### 🧠 Feature Dependence Analysis")
                                        if f_importance_dict:
                                            # Convert dict to Dataframe for Plotly Express
                                            df_importance = pd.DataFrame(
                                                list(f_importance_dict.items()), 
                                                columns=['Feature', 'Importance (Normalized)']
                                            )
                                            # Plotly bar chart
                                            fig_imp = px.bar(
                                                df_importance.head(10), # Show top 10 for clarity
                                                x='Importance (Normalized)', 
                                                y='Feature', 
                                                orientation='h',
                                                title='Winning factors driving model output (Feature Importance)',
                                                template="plotly_dark",
                                                color='Importance (Normalized)',
                                                color_continuous_scale='Viridis'
                                            )
                                            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort chart
                                            st.plotly_chart(fig_imp, use_container_width=True)
                                        else:
                                            st.info("Feature Dependence analysis not available for this run.")

                                        st.markdown("---")
                                        
                                        # (D-2) 🟢 MISSING SCATTER PLOT WAPAS AA GAYA
                                        st.markdown("### 🎯 Actual vs Predicted Analysis")
                                        actual_targets = data.get("actual_targets", [])
                                        predicted_targets = data.get("predicted_targets", [])
                                        
                                        if actual_targets and predicted_targets:
                                            df_preds = pd.DataFrame({'Actual': actual_targets, 'Predicted': predicted_targets})
                                            fig_scatter = px.scatter(
                                                df_preds, x='Actual', y='Predicted', 
                                                title="Prediction Accuracy (Closer to line = Better)",
                                                template="plotly_dark",
                                                color_discrete_sequence=['#00BCD4']
                                            )
                                            min_val = min(min(actual_targets), min(predicted_targets))
                                            max_val = max(max(actual_targets), max(predicted_targets))
                                            fig_scatter.add_shape(
                                                type="line", line=dict(dash='dash', color='white'),
                                                x0=min_val, y0=min_val, x1=max_val, y1=max_val
                                            )
                                            st.plotly_chart(fig_scatter, use_container_width=True)
                                        else:
                                            st.info("Scatter plot data not available for this run.")
                                            
                                        st.markdown("---")

                                        # (E) COMBINED GRAPH (Learning Curve)
                                        st.markdown("### 📈 Training vs Validation (Overfitting Check)")
                                        fig_curve = make_subplots(specs=[[{"secondary_y": True}]])
                                        fig_curve.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train Loss (MSE)', line=dict(color='#4CAF50', width=2)),secondary_y=False)
                                        fig_curve.add_trace(go.Scatter(x=epochs, y=test_rmses, mode='lines', name='Test RMSE', line=dict(color='#FF9800', width=2)),secondary_y=True)
                                        
                                        fig_curve.update_layout(title="Model Learning Curve", template="plotly_dark", height=450, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                                        fig_curve.update_yaxes(title_text="Training Loss", secondary_y=False)
                                        fig_curve.update_yaxes(title_text="Validation RMSE", secondary_y=True)
                                        st.plotly_chart(fig_curve, use_container_width=True)
                                        
                                        # (F) Detailed metrics table
                                        st.markdown("---")
                                        st.markdown("#### 📊 Detailed Metrics Table")
                                        table_data = [{"Epoch": int(epochs[i]) if i < len(epochs) else i, "Train Loss": f"{train_losses[i]:.6f}" if i < len(train_losses) else "N/A", "Test RMSE": f"{test_rmses[i]:.6f}" if i < len(test_rmses) else "N/A"} for i in range(len(epochs))]
                                        st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=350)
                                        
                                        # Downloads (Simplified original code downloads CSV of detailed table)
                                        csv = pd.DataFrame(table_data).to_csv(index=False)
                                        st.download_button(label="📥 Download Metrics as CSV", data=csv, file_name=f"metrics_{sub['target_column'].replace(' ', '_')}.csv", mime="text/csv")
                                        
                                        if sub.get("status") in ["training", "pending", "in_progress"]:
                                            time.sleep(2.5)
                                            st.rerun()
                                    else:
                                        st.warning("⚠️ Metrics arrays are empty in DB.")
                                        
                                else:
                                    st.error(f"❌ Metrics Error: {metrics_res.status_code} - {metrics_res.text}")
                            
                            except requests.exceptions.Timeout:
                                st.error("⏱️ live metrics request timed out")
                            except Exception as e:
                                st.error(f"❌ Error fetching metrics: {str(e)}")
            else:
                st.error(f"Failed to load submissions: {res.status_code}")
        
        except Exception as e:
            st.error(f"Tab 3 Error: {str(e)}")