import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Network Guard (Production)", layout="wide")

# --- HEADER ---
st.title("🛡️ AI-Based NIDS: Production Mode")
st.markdown("""
This system trains on the **CIC-IDS2017 Real-World Dataset**.
Focus: Detecting **DDoS Attacks** (Distributed Denial of Service).
""")

# --- STEP 1: LOAD REAL DATA ---
@st.cache_data # Caches data so it doesn't reload every time you click a button
def load_data():
    # 1. Load the CSV file
    # We use a subset (nrows) to ensure it runs fast on your laptop. 
    # Remove 'nrows=20000' to train on the full file (might be slow).
    try:
        df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=20000)
    except FileNotFoundError:
        return None

    # 2. Clean Column Names (Remove hidden spaces)
    df.columns = df.columns.str.strip()

    # 3. Select Key Features for the Demo
    # The dataset has 70+ columns. We select the 3 most important ones for the demo UI.
    # 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets'
    selected_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Label']
    df = df[selected_features]

    # 4. Handle Missing/Infinite Values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 5. Encode Labels (Text -> Numbers)
    # BENIGN = 0 (Normal), DDoS = 1 (Attack)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    return df

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🔧 Production Controls")

if st.sidebar.button("Train on Real Data"):
    with st.spinner('Loading CIC-IDS2017 Dataset and Training...'):
        
        df = load_data()
        
        if df is None:
            st.error("❌ File not found! Please make sure 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' is in your folder.")
        else:
            # Prepare Data
            X = df.drop('Label', axis=1) # Features
            y = df['Label']              # Target

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Model
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Save to Session
            st.session_state['prod_model'] = clf
            st.session_state['prod_acc'] = acc
            st.success(f"Training Complete! Loaded {len(df)} real packets.")

# --- MAIN DASHBOARD ---

if 'prod_acc' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model Accuracy (Real Data)", value=f"{st.session_state['prod_acc']*100:.2f}%")
    with col2:
        st.info("System is monitoring for DDoS patterns.")
    
    st.markdown("---")

    # --- LIVE TRAFFIC INTERFACE ---
    st.subheader("🔎 Traffic Inspector")
    st.write("Input network values to test against the CIC-IDS2017 trained model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        # Real data 'Flow Duration' is usually in microseconds
        flow_duration = st.number_input("Flow Duration (microseconds)", min_value=0, value=100000)
    with c2:
        total_fwd = st.number_input("Total Forward Packets", min_value=0, value=5)
    with c3:
        total_bwd = st.number_input("Total Backward Packets", min_value=0, value=0)

    if st.button("Analyze Traffic"):
        # Match the input structure to the training data
        input_data = [[flow_duration, total_fwd, total_bwd]]
        
        prediction = st.session_state['prod_model'].predict(input_data)[0]
        
        if prediction == 1:
            st.error("🚨 ALERT: DDoS ATTACK DETECTED")
            st.write("The traffic patterns match the DDoS signatures in the training dataset.")
        else:
            st.success("✅ Normal Traffic")
            st.write("Traffic behavior appears benign.")

else:
    st.warning("👈 Download the dataset and click 'Train on Real Data' to begin.")

# --- DEBUGGING TOOL ---
if st.checkbox("Show Dataset Sample"):
    try:
        df_sample = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=5)
        st.write(df_sample.head())
    except:
        st.write("Dataset not found.")