import streamlit as st
import requests
import pandas as pd
import io
import plotly.graph_objects as go  
import time

import os
API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NexusOps Hub", page_icon="⚡", layout="wide")


st.markdown("""
<style>
    /* Premium Minimalist Theme Variables */
    :root {
        --primary: #00D9FF;
        --secondary: #5390FF;
        --light-text: #E4E6EB;
        --subtle-text: #9299A4;
    }

    /* Clean Dark Canvas (Center Body) */
    .stApp {
        background: radial-gradient(circle at top, #141B29 0%, #0C1017 60%, #06080C 100%) !important;
        color: var(--light-text);
    }

    /* Elegantly Balanced Hero Title */
    .main-header {
        font-family: 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(90deg, #00D9FF 0%, #5390FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.8rem !important;
        font-weight: 800;
        letter-spacing: -2px;
        margin-top: 2rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-header {
        text-align: center;
        color: var(--subtle-text);
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
        letter-spacing: -0.2px;
    }

    /* Minimalist Tech Stack Subtext */
    .tech-subtext {
        text-align: center;
        color: #00D9FF;
        font-size: 0.9rem;
        font-family: monospace;
        letter-spacing: 1px;
        margin-bottom: 3.5rem;
        opacity: 0.8;
    }

    /* Ultra-Clean Sleek Cards (No heavy borders) */
    .feature-card {
        background: rgba(22, 27, 34, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(0, 217, 255, 0.3);
        background: rgba(22, 27, 34, 0.7);
        transform: translateY(-2px);
    }

    /* Premium Modern Main Page Buttons */
    .stButton>button {
        background: #1f6feb !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background: #2f81f7 !important;
        box-shadow: 0 0 15px rgba(31, 111, 235, 0.4);
    }

    /* ==================== HIGH-CONTRAST PREMIUM SIDEBAR ==================== */
    /* Graphite Gray Background to visually separate from main body canvas */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.4);
    }

    /* Premium Sleek Segmented Switcher */
    [data-testid="stSidebar"] .stRadio > div {
        background: #0D1117 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        font-family: 'Segoe UI', system-ui, sans-serif !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #8B949E !important;
    }

    /* Sharp High-Contrast Active Tab State */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
        background: #21262D !important;
        color: #00D9FF !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
    }

    /* Clean Matte-Dark Input Fields (No Monospace/Robotic Fonts) */
    [data-testid="stSidebar"] .stTextInput input {
        background-color: #0D1117 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        color: #F0F6FC !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
    }

    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #00D9FF !important;
        box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.15) !important;
    }

    /* Cyber-Cyan Premium Action Button */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #00B4D8 0%, #00D9FF 100%) !important;
        color: #0A0D14 !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: -0.2px !important;
        border-radius: 8px !important;
        padding: 12px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0, 217, 255, 0.15) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #00D9FF 0%, #46E5FF 100%) !important;
        box-shadow: 0 6px 18px rgba(0, 217, 255, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Elegant Sidebar Section Headings */
    .sidebar-subheading {
        color: #F0F6FC;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Global Status Components */
    .stSuccess { background-color: rgba(76, 175, 80, 0.05) !important; border-left: 3px solid #4CAF50; }
    .stError { background-color: rgba(255, 107, 107, 0.05) !important; border-left: 3px solid #FF6B6B; }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        margin: 3.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


from streamlit_cookies_controller import CookieController
import time

controller = CookieController()

cookie_token = controller.get("nexus_token")

if cookie_token and st.session_state.get("access_token") is None:
    st.session_state.access_token = cookie_token
elif not cookie_token and "access_token" not in st.session_state:
    st.session_state.access_token = None

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}"}

# ======================== SIDEBAR AUTH ========================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-top: 1rem; margin-bottom: 2rem;'>
        <h2 style='color: #00D9FF; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px;'>⚡ NexusOps</h2>
        <p style='color: #8B949E; margin-top: 2px; font-size: 0.85rem;'>Control Plane v1.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.access_token:
        st.success("🔒 Control Plane Active")
        if st.button("🚪 Terminate Session", use_container_width=True):
            
            st.session_state.access_token = None
            controller.remove("nexus_token")
            time.sleep(0.5)
            st.rerun()
    else:
       
        auth_mode = st.radio("Choose Action", ["🔐 Login", "📝 Register"], horizontal=True, label_visibility="collapsed")
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        
        if auth_mode == "🔐 Login":
            st.markdown("<p class='sidebar-subheading'>Authorize Operator Credentials</p>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Operator identity ID", label_visibility="collapsed", key="login_user")
            st.markdown("<div style='margin-bottom: 0.4rem;'></div>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Access token security key", label_visibility="collapsed", key="login_pass")
            st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)
            
            if st.button("Initialize Gateway Control →", use_container_width=True):
                res = requests.post(
                    f"{API_URL}/auth/login",
                    data={"username": username, "password": password}
                )
                if res.status_code == 200:
                    new_token = res.json()["access_token"]
                    
                   
                    st.session_state.access_token = new_token
                    controller.set("nexus_token", new_token, max_age=604800) 
                    
                    st.success("✅ Initialized.")
                    time.sleep(1) 
                    st.rerun()
                else:
                    st.error("❌ Refused by auth cluster.")
        
        elif auth_mode == "📝 Register":
            st.markdown("<p class='sidebar-subheading'>Provision New Operational Node</p>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Operator node handle", label_visibility="collapsed", key="reg_user")
            st.markdown("<div style='margin-bottom: 0.4rem;'></div>", unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="Corporate email registration", label_visibility="collapsed", key="reg_email")
            st.markdown("<div style='margin-bottom: 0.4rem;'></div>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Master cryptographic phrase", label_visibility="collapsed", key="reg_pass")
            st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)
            
            if st.button("Compile Operator Node 🛠️", use_container_width=True):
                res = requests.post(
                    f"{API_URL}/auth/register",
                    json={"username": username, "email": email, "password": password}
                )
                if res.status_code == 201:
                    st.success("✅ Compiled! Ready for Login.")
                else:
                    try:
                        error_msg = res.json().get("detail", "Provisioning rejected.")
                    except ValueError:
                        error_msg = f"Registration failed."
                    st.error(error_msg)

# ======================== MAIN APP INTERFACE ========================
st.markdown("<h1 class='main-header'>⚡ NexusOps</h1>", unsafe_allow_html=True)

if not st.session_state.access_token:
    st.markdown("<p class='sub-header'>The Asynchronous AutoML Engine for Enterprise Deep Learning</p>", unsafe_allow_html=True)
    st.markdown("<p class='tech-subtext'>FASTAPI • CELERY • REDIS • PYTORCH • OPTUNA • MONGO</p>", unsafe_allow_html=True)
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.markdown("""
        <div class='feature-card'>
            <span style='font-size: 2.2rem;'>📥</span>
            <h4 style='color: #00D9FF; margin-top: 0.8rem; margin-bottom: 0.5rem;'>Ingest Payloads</h4>
            <p style='color: #9299A4; font-size: 0.9rem; line-height: 1.4; margin: 0;'>
                Drop raw tabular datasets. Instant automated schema verification and validation staging.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_step2:
        st.markdown("""
        <div class='feature-card'>
            <span style='font-size: 2.2rem;'>⚙️</span>
            <h4 style='color: #5390FF; margin-top: 0.8rem; margin-bottom: 0.5rem;'>Async AutoML</h4>
            <p style='color: #9299A4; font-size: 0.9rem; line-height: 1.4; margin: 0;'>
                Background neural architecture searches via Celery & Redis. Zero interface latency.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_step3:
        st.markdown("""
        <div class='feature-card'>
            <span style='font-size: 2.2rem;'>📦</span>
            <h4 style='color: #4CAF50; margin-top: 0.8rem; margin-bottom: 0.5rem;'>Export Artifacts</h4>
            <p style='color: #9299A4; font-size: 0.9rem; line-height: 1.4; margin: 0;'>
                Monitor live telemetry metrics and download production-ready <code>.pth</code> weights and <code>.json</code> configs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(0, 217, 255, 0.03); border: 1px solid rgba(0, 217, 255, 0.1); border-radius: 12px; padding: 1.8rem; text-align: center; max-width: 800px; margin: 0 auto;'>
        <h4 style='margin: 0; color: #00D9FF; font-weight: 600; letter-spacing: -0.2px;'>🔒 Operational Control Plane Staged</h4>
        <p style='margin: 0.6rem 0 0 0; color: #9299A4; font-size: 0.95rem; line-height: 1.4;'>
            To trigger new automated pipeline runs, monitor live clusters, or infer against trained models, please authenticate using the Operator Workspace controls in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("<p class='sub-header'>Cluster Control Plane Active • Workspace Ready</p>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🚀 Submit New Request", "📂 My Submissions", "📊 Live Metrics"])
    
    with tab1:
        st.info("Form submission controls go here.")
    with tab2:
        st.info("Submissions tracker loop goes here.")
    with tab3:
        st.info("Live data telemetry components go here.")
        
        
    # ======================== TAB 1: SUBMISSION FORM ========================
    with tab1:
        st.markdown("### 📤 Upload Your Dataset & Requirements")
        st.write("Fill out the details below to define the architecture and expectations you have from the model.")
        
        import streamlit as st
        import requests
        import re  

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
                            
                           
                            formatted_target = target_column.strip().lower()
                            formatted_target = re.sub(r'\s+', '_', formatted_target)
                            
                            files = {"dataset": (dataset_file.name, dataset_file, dataset_file.type)}
                            
                            
                            data = {
                                "target_column": formatted_target, 
                                "use_case": use_case,
                                "requirement": requirement
                            }
                            
                            try:
                               
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
                        
                        sub_id = str(sub.get('_id', sub.get('id', ''))) 
                        status = sub.get('status', 'pending')
                        
                        with st.expander(f"Dataset: {sub.get('target_column', 'Unknown')} | Status: {status.upper()}"):
                            
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
                            
                            st.divider() 
                            
                            if status in ['pending', 'failed']:
  
                                st.markdown("<p style='color: #00D9FF; font-size: 0.9rem; margin-bottom: 0;'><b>⚙️ Distributed Engine Settings</b></p>", unsafe_allow_html=True)
                                col_t, col_w = st.columns(2)
                                with col_t:
                                    user_trials = st.slider("🧠 Trials", 5, 100, 20, step=5, key=f"trials_{sub_id}")
                                with col_w:
                                    user_workers = st.slider("⚡ Workers", 1, 16, 4, step=1, key=f"workers_{sub_id}")

                                if st.button("🚀 Train Model", key=f"train_{sub_id}"):
                                    
                                    payload = {
                                        "n_trials": user_trials,
                                        "n_workers": user_workers
                                    }
                                    
                                    
                                    t_res = requests.post(
                                        f"{API_URL}/submit/{sub_id}/train", 
                                        headers=get_headers(),
                                        json=payload  
                                    )
                                    
                                    if t_res.status_code == 200:
                                        st.toast("🚀 Training Request Sent!")
                                        
                                        progress_bar = st.progress(5, text="Starting MLOps Pipeline...")
                                        
                                        # Naye placeholders Live Timer aur Messages ke liye
                                        timer_placeholder = st.empty()
                                        status_message = st.empty()
                                        
                                        import time
                                        is_training = True
                                        start_time = time.time() # Training ka start time note kiya
                                        last_api_call_time = 0   # Track karenge aakhiri baar API kab bulai
                                        
                                        # Backend states track karne ke liye variables
                                        current_status = "pending"
                                        epochs_done = 0
                                        
                                        while is_training:
                                            # 1. LIVE STOPWATCH & TIME CALCULATION (Har loop me update hoga)
                                            elapsed_seconds = int(time.time() - start_time)
                                            mins, secs = divmod(elapsed_seconds, 60)
                                            timer_placeholder.markdown(f"### ⏱️ Time Elapsed: **{mins}m {secs}s**")
                                            
                                            # 2. SMART INTERACTIVE MESSAGES (Time aur Status ke hisaab se)
                                            if current_status == "pending":
                                                progress_bar.progress(15, text="⏳ Step 1/3: Queued in Server...")
                                                status_message.info("Waiting for Celery worker to pick up the task...")
                                                
                                            elif current_status in ["in_progress", "training"]:
                                                if epochs_done == 0:
                                                    progress_bar.progress(40, text="⚙️ Step 2/3: Running Optuna Hyperparameter Trials...")
                                                    
                                                    # TIME-AWARE OPTUNA MESSAGES (The Magic Trick)
                                                    if elapsed_seconds < 60:
                                                        status_message.warning("Testing multiple AI architectures to find the best one. (Usually takes 1-2 mins)...")
                                                    elif elapsed_seconds < 150: # 2.5 Mins
                                                        status_message.warning("Still testing architectures. Finding the global minimum loss...")
                                                    elif elapsed_seconds < 300: # 5 Mins
                                                        status_message.info("💡 This seems to be a heavy dataset! The Optuna engine is doing deep analysis. Grab a coffee ☕...")
                                                    elif elapsed_seconds < 600: # 10 Mins
                                                        status_message.info("🔥 We are working hard! Searching through hundreds of hidden layer combinations. Please hold tight...")
                                                    else:
                                                        status_message.info("🐢 The dataset is massive! Workers are crunching the final layers. Thanks for your patience...")
                                                else:
                                                    progress_bar.progress(75, text=f"🧠 Step 3/3: Final Training (Epoch {epochs_done} done)...")
                                                    status_message.info(f"Best architecture found! Training final production model... (Epoch {epochs_done} recorded)")
                                                    
                                            elif current_status == "completed":
                                                progress_bar.progress(100, text="✅ Training Completely Finished!")
                                                timer_placeholder.empty() # Timer hata do completion pe
                                                status_message.success("🎉 Model trained successfully! Switch to the 'Training Metrics' tab (Tab 3) to view AI Insights and Graphs.")
                                                st.balloons() 
                                                is_training = False
                                                time.sleep(3) 
                                                st.rerun() 
                                                
                                            elif current_status in ["failed", "error"]:
                                                progress_bar.empty()
                                                timer_placeholder.empty()
                                                status_message.error("❌ Training failed! Please check backend terminal logs.")
                                                is_training = False

                                            # 3. BACKEND API CALL (Har 5 second me ek baar karenge)
                                            # 3. BACKEND API CALL (Har 5 second me ek baar karenge)
                                            if is_training and (time.time() - last_api_call_time) >= 5:
                                                try:
                                                    # 🔥 MAGIC FIX: Added 'timeout=2' to prevent UI freezing!
                                                    check_res = requests.get(
                                                        f"{API_URL}/submit/{sub_id}/live-metrics", 
                                                        headers=get_headers(),
                                                        timeout=2
                                                    )
                                                    if check_res.status_code == 200:
                                                        data = check_res.json()
                                                        current_status = data.get("status", "pending")
                                                        epochs_done = len(data.get("epochs", []))
                                                
                                                except requests.exceptions.Timeout:
                                                    # Agar backend busy hai, toh UI freeze nahi hoga
                                                    pass 
                                                except Exception as e:
                                                    pass # Baaki errors chupchap ignore karenge
                                                
                                                last_api_call_time = time.time() # Update API call time

                                            # 4. SLEEP FOR 1 SECOND (Jisse Stopwatch smooth chale)
                                            if is_training:
                                                time.sleep(1)
                                                
                                    else:
                                        st.error(f"Error starting training: {t_res.text}")

                            elif status == 'training':
                                st.info("⏳ Model is currently training... MLflow metrics will be available once training completes.")
                                    
                            elif status == 'completed':
                                st.success("✅ Training Completed - Model is ready!")
                                
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                
                                if st.button("📊 View Live AI Dashboard", key=f"mlflow_{sub_id}", use_container_width=True):
                                    st.success("👆 Model selected! Please click on 'Tab 3' above to view insights.")
                                    
                            with col2:
                                
                                if st.session_state.get(f"json_ready_{sub_id}"):
                                    
                                   
                                    config_data = st.session_state[f"json_data_{sub_id}"]
                                    import json
                                    json_string = json.dumps(config_data, indent=4)
                                    target_name = sub.get('target_col', 'model').replace(' ', '_').lower() 
                                    
                                    st.download_button(
                                        label="📄 Download Config (.json)",
                                        data=json_string,
                                        file_name=f"{target_name}_config.json",
                                        mime='application/json',
                                        use_container_width=True,
                                        key=f"dl_json_ready_{sub_id}"
                                    )
                                    
                                  
                                    pth_url = f"{API_URL}/submit/{sub_id}/model-file"
                                    st.markdown(f'''
                                        <a href="{pth_url}" download style="text-decoration: none;">
                                            <button style="width:100%; border:1px solid #1f77b4; color:#1f77b4; background-color:transparent; padding:0.4rem; border-radius:4px; cursor:pointer; margin-top:2px;">
                                                💾 Download Weights (.pth)
                                            </button>
                                        </a>
                                    ''', unsafe_allow_html=True)
                                    
                                else:
                                    
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
        

        st.markdown("### 📊 Training Metrics Analysis")
        st.write("View detailed metrics and strategic insights from completed training.")
        
        try:
            
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px 
            import pandas as pd
            import time

            
            res = requests.get(f"{API_URL}/submit/", headers=get_headers())
            
            if res.status_code == 200:
                submissions = res.json()
                completed = [s for s in submissions if s.get("status") == "completed"]
                
                if not completed:
                    st.info("📭 No completed trainings found yet.")
                else:
                    
                    selected_sub = st.selectbox(
                        "Select Model to View Performance", 
                        options=completed, 
                        format_func=lambda x: f"{x.get('target_column', 'model')} | {x.get('created_at', 'Date N/A')[:19].replace('T', ' ')} (ID: {str(x.get('_id', ''))[-5:]})",
                        key="metrics_select"
                    )
                    
                    if selected_sub:
                        sub = selected_sub
                        sub_id = str(sub.get("_id", ""))
                        
                        
                        st.markdown(f"#### 🎯 Forecast Model for: `{sub['target_column']}`")
                        
                       
                        status_col, target_col, mlflow_col = st.columns(3)
                        status_col.metric("Workflow Status", "✅ Completed")
                        target_col.metric("Target Column", sub['target_column'])
                        mlflow_col.metric("MLflow Run ID", sub.get("mlflow_run_id", "N/A"))
                        
                        st.markdown("---")
                        
                        
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
                                    
                                    
                                    epochs = data.get("epochs", [])
                                    train_losses = data.get("train_losses", [])
                                    test_rmses = data.get("test_rmses", [])
                                    test_maes = data.get("test_maes", []) 
                                    test_r2s = data.get("test_r2s", [])   
                                    final_score = data.get("final_score")
                                    
                                    
                                    exec_stats = data.get("execution_stats", {})
                                    best_params = data.get("best_params", {})
                                    
                                    
                                    f_importance_dict = data.get("feature_importance", {})
                                    llm_summary_paragraph = data.get("llm_executive_summary", "")

                                    st.success(f"✅ Loaded {len(epochs)} epochs of metrics.")
                                    
                                    if epochs and train_losses and test_rmses:
                                        
                                        
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

                                        
                                        st.markdown("### 🤖 Model Executive Summary & AI Insights")
                                        if llm_summary_paragraph:
                                            
                                            st.info(f"👉 {llm_summary_paragraph}")
                                        else:
                                            st.info("Insights for this run have not been generated yet. (They are only available on newly trained models).")
                                            
                                        st.markdown("---")

                                        
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

                                        
                                        st.markdown("### 🧠 Feature Dependence Analysis")
                                        if f_importance_dict:
                                            
                                            df_importance = pd.DataFrame(
                                                list(f_importance_dict.items()), 
                                                columns=['Feature', 'Importance (Normalized)']
                                            )
                                            
                                            fig_imp = px.bar(
                                                df_importance.head(10), 
                                                x='Importance (Normalized)', 
                                                y='Feature', 
                                                orientation='h',
                                                title='Winning factors driving model output (Feature Importance)',
                                                template="plotly_dark",
                                                color='Importance (Normalized)',
                                                color_continuous_scale='Viridis'
                                            )
                                            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}) 
                                            st.plotly_chart(fig_imp, use_container_width=True)
                                        else:
                                            st.info("Feature Dependence analysis not available for this run.")

                                        st.markdown("---")
                                        
                                       
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

                                        st.markdown("### 📈 Training vs Validation (Overfitting Check)")
                                        fig_curve = make_subplots(specs=[[{"secondary_y": True}]])
                                        fig_curve.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train Loss (MSE)', line=dict(color='#4CAF50', width=2)),secondary_y=False)
                                        fig_curve.add_trace(go.Scatter(x=epochs, y=test_rmses, mode='lines', name='Test RMSE', line=dict(color='#FF9800', width=2)),secondary_y=True)
                                        
                                        fig_curve.update_layout(title="Model Learning Curve", template="plotly_dark", height=450, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                                        fig_curve.update_yaxes(title_text="Training Loss", secondary_y=False)
                                        fig_curve.update_yaxes(title_text="Validation RMSE", secondary_y=True)
                                        st.plotly_chart(fig_curve, use_container_width=True)
                                        
                                        
                                        st.markdown("---")
                                        st.markdown("#### 📊 Detailed Metrics Table")
                                        table_data = [{"Epoch": int(epochs[i]) if i < len(epochs) else i, "Train Loss": f"{train_losses[i]:.6f}" if i < len(train_losses) else "N/A", "Test RMSE": f"{test_rmses[i]:.6f}" if i < len(test_rmses) else "N/A"} for i in range(len(epochs))]
                                        st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=350)
                                        
                                    
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