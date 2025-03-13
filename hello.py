import streamlit as st
import cv2
import glob 
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import uuid
import time
import requests
import os
import pandas as pd
import numpy as np
# Initialize session state
if 'capture' not in st.session_state:
    st.session_state.capture = False
if 'unrecognized_face' not in st.session_state:
    st.session_state.unrecognized_face = None
if 'show_registration' not in st.session_state:
    st.session_state.show_registration = False
if 'security_recognized' not in st.session_state:
    st.session_state.security_recognized = False
if 'security_setup' not in st.session_state:
    st.session_state.security_setup = False
if 'full_auth' not in st.session_state:
    st.session_state.full_auth = False
if 'pending_request' not in st.session_state:
    st.session_state.pending_request = None
# Initialize counts in session state
def initialize_session_state():
    defaults = {
        'total_blacklisted_users': 0,
        'visitors_blocked': 0,
        'registered_users_blocked': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()
# Configuration
ENCODINGS_PATH = "encodings/face_encodings.pkl"
SECURITY_NAME = "Security"
LOGS_DIR = Path("Logs")
SECURITY_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "rahmathmohd1654@gmail.com"  # Replace with your Gmail
SMTP_PASSWORD = "hxuk kloj vhdb xooh"    # Replace with your Gmail app password
SECURITY_EMAIL = "rahmathmohd1654@gmail.com"  # Replace with your Gmail

# FastAPI Backend URL (updated with your IPv4 address)
FASTAPI_URL = "https://sad-pvly.onrender.com"

arcface = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' is a pre-trained ArcFace model
arcface.prepare(ctx_id=1) 
# Create directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)
Path("encodings").mkdir(exist_ok=True)
BLACKLIST_IMAGES_DIR = Path("blacklist_images")
BLACKLIST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def cosine_similarity(embedding1, embedding2):
    # Ensure both embeddings are numpy arrays
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def load_encodings():
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            # Ensure blacklist_metadata exists
            if "blacklist_metadata" not in data:
                data["blacklist_metadata"] = []
            if "image_paths" not in data:
                data["image_paths"] = []  
            return data
    except (FileNotFoundError, EOFError):
        # Initialize data with all required keys
        return {
            "names": [],
            "phones": [],
            "emails": [],
            "encodings": [],
            "blacklist": [],
            "blacklist_metadata": [],
            "image_paths": []   # Initialize blacklist_metadata
        }

def save_image(image, name, phone, is_registered):
    today = datetime.now().strftime("%Y-%m-%d")
    base_dir = "user_images" if is_registered else "visitor_images"
    image_dir = Path(base_dir) / today
    image_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{name}_{phone}.jpg"
    image_path = image_dir / filename
    cv2.imwrite(str(image_path), image)
    return image_path

def save_features(name, phone, embedding):
    feature_path = Path("encodings/feature.xlsx")
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate professional feature names
    num_features = len(embedding)
    columns = ["Name", "Phone"] + [f"ArcFace_Embedding_{i:03d}" for i in range(1, num_features+1)]
    
    # Create a DataFrame for the new entry
    new_row = [name, phone] + [f"{x:.6f}" for x in embedding]
    new_df = pd.DataFrame([new_row], columns=columns)

    # Append to existing Excel file or create new
    if feature_path.exists():
        existing_df = pd.read_excel(feature_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    # Save to Excel
    updated_df.to_excel(feature_path, index=False)
def save_encodings(data):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)
def update_blacklist_counts(blacklist_metadata):
    """Update session state blacklist counts based on metadata."""
    st.session_state.total_blacklisted_users = len(blacklist_metadata)
    st.session_state.visitors_blocked = sum(1 for x in blacklist_metadata if x.get("type") == "visitor")
    st.session_state.registered_users_blocked = (
        st.session_state.total_blacklisted_users - st.session_state.visitors_blocked
    )
def detect_and_encode(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = arcface.get(rgb_image)
    return faces[0].embedding if faces else None


def log_visit(name, phone, action="visit", approver="system"):
    now = datetime.now()
    log_path = LOGS_DIR / now.strftime("%Y-%m-%d")
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "logs.xlsx"

    new_entry = pd.DataFrame([[action, name, phone, approver, now.isoformat()]],
                             columns=["Action Type", "Visitor Name", "Phone Number", "Authorized By", "Timestamp"])

    if log_file.exists():
        existing_logs = pd.read_excel(log_file)
        updated_logs = pd.concat([existing_logs, new_entry], ignore_index=True)
    else:
        updated_logs = new_entry

    updated_logs.to_excel(log_file, index=False)

def get_registered_residents():
    data = load_encodings()
    return [(name, phone, email) for name, phone, email in zip(data["names"], data["phones"], data["emails"]) 
            if name != SECURITY_NAME]

def send_approval_email(visitor_name, visitor_phone, resident_email, request_id, image_path):
    approve_url = f"{FASTAPI_URL}/approve/{request_id}"
    deny_url = f"{FASTAPI_URL}/deny/{request_id}"
    blacklist_url = f"{FASTAPI_URL}/blacklist/{request_id}"
    
    message = MIMEMultipart()
    message["From"] = SECURITY_EMAIL
    message["To"] = resident_email
    message["Subject"] = "🚨 Visitor Approval Request"
    
    # Attach the image
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={image_path.name}",
        )
        message.attach(part)
    
    # Email body with HTML forms
    body = f"""
    <html>
      <body>
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h3 style="color: #2c3e50;">Visitor Approval Request</h3>
          <p>Visitor Details:</p>
          <ul>
            <li><strong>Name:</strong> {visitor_name}</li>
            <li><strong>Phone:</strong> {visitor_phone}</li>
          </ul>
          
          <p style="margin: 20px 0;">Please choose an action:</p>
          
          <table width="100%" cellspacing="0" cellpadding="0">
            <tr>
              <td>
                <table cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="border-radius: 5px; background-color: #4CAF50; padding: 12px 24px;">
                      <a href="{approve_url}" style="color: white; text-decoration: none; display: inline-block;">
                        ✅ Approve
                      </a>
                    </td>
                    <td width="20"></td>
                    <td style="border-radius: 5px; background-color: #f44336; padding: 12px 24px;">
                      <a href="{deny_url}" style="color: white; text-decoration: none; display: inline-block;">
                        ❌ Deny
                      </a>
                    </td>
                    <td width="20"></td>
                    <td style="border-radius: 5px; background-color: #000000; padding: 12px 24px;">
                      <a href="{blacklist_url}" style="color: white; text-decoration: none; display: inline-block;">
                        🚫 Blacklist
                      </a>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
          </table>
        </div>
      </body>
    </html>
    """
    
    message.attach(MIMEText(body, "html"))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SECURITY_EMAIL, resident_email, message.as_string())
        st.success("Approval email sent successfully!")
    except Exception as e:
        st.error("Failed to send approval email")
        st.error(str(e))

def create_pending_request(visitor_name, visitor_phone, resident_email, request_id):
    url = f"{FASTAPI_URL}/request/{request_id}"
    payload = {
        "visitor_name": visitor_name,
        "visitor_phone": visitor_phone,
        "resident_email": resident_email,
        "status": "pending"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True
    else:
        st.error("Failed to create pending request in backend")
        return False

def check_approval_status(request_id):
    try:
        response = requests.get(f"{FASTAPI_URL}/status/{request_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        st.error("Connection to backend failed")
        return None

def security_gate():
    data = load_encodings()
    if SECURITY_NAME in data["names"]:
        return

    st.title("🔒 Security Initialization")
    st.warning("Security profile not configured. First-time setup required.")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        password = st.text_input("Enter setup password:", type="password")
        if st.button("Initialize Security System"):
            if hashlib.sha256(password.encode()).hexdigest() == SECURITY_PASSWORD_HASH:
                st.session_state.security_setup = True
            else:
                st.error("Invalid setup password")
    with col2:
        st.markdown("### Setup Instructions")
        st.write("1. Obtain setup password from system administrator")
        st.write("2. Ensure good lighting and face visibility")
        st.write("3. Position face in the camera frame")
        st.write("4. Click 'Capture Security Profile' after password validation")

    if st.session_state.get("security_setup"):
        img_file = st.camera_input("Capture security face (frontal view)")
        if img_file:
            image = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            face_encoding = detect_and_encode(image)
            if face_encoding is not None:
                data["names"].append(SECURITY_NAME)
                data["phones"].append("SECURITY ")
                data["emails"].append(SECURITY_EMAIL)
                data["encodings"].append(face_encoding)
                save_encodings(data)
                st.success("Security profile configured successfully!")
                st.rerun()
            else:
                st.error("No face detected - try again")
    st.stop()

def security_dashboard():
    st.title("🔐 Security Dashboard")
    data = load_encodings()
    with st.expander("➕ Register New User"):
        with st.form("registration_form"):
            name = st.text_input("Full Name", key="reg_name")
            phone = st.text_input("Phone Number", key="reg_phone")
            email = st.text_input("Email Address", key="reg_email")
            img_file = st.camera_input("Take registration photo", key="reg_cam")
            
            if st.form_submit_button("Register User"):
                if name and phone and email and img_file:
                    # Check for existing user
                    data = load_encodings()
                    duplicate = False
                    face_duplicate = False
                    
                    # Convert captured image to encoding
                    image = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    new_encoding = detect_and_encode(image)
                    
                    if new_encoding is None:
                        st.error("No face detected in the image. Please ensure proper lighting and face visibility.")
                        st.stop()
                    if any(float(cosine_similarity(new_encoding, enc)) > 0.6 for enc, phone in zip(data["encodings"], data["phones"]) if phone == "SECURITY "):
                        st.error("🛑 Nice try, Mr. Security! You already have VIP access. No need to register again. Go patrol the gates! 😆")
                        st.stop()
                    
                    # Check against existing encodings (excluding security)
                    for i in range(len(data["names"])):
                        if data["names"][i] == SECURITY_NAME:
                            continue
                        
                        # Check personal details
                        if (data["names"][i] == name or 
                            data["phones"][i] == phone or 
                            data["emails"][i] == email):
                            duplicate = True
                            break
                        
                        match = cosine_similarity(data["encodings"][i], new_encoding) > 0.6
                        if match:
                            face_duplicate = True
                            existing_name = data["names"][i]
                            break  # Stop checking once a match is found

                    
                    if duplicate:
                        st.error("❌ User already exists with same name, phone, or email!")
                    elif face_duplicate:
                        st.error(f"❌ Face already registered as {existing_name}!")
                    else:
                        image_path = save_image(image, name, phone, is_registered=True)
                        save_features(name, phone, new_encoding) 
                        data["names"].append(name)
                        data["phones"].append(phone)
                        data["emails"].append(email)
                        data["encodings"].append(new_encoding)
                        data["image_paths"].append(str(image_path))
                        save_encodings(data)
                        log_visit(name, phone, "registered", "Security")
                        st.success("✅ User registered successfully!")
                        st.rerun()
                else:
                    st.error("Please fill all fields and capture a photo")

    with st.expander("🚫 Manage Blacklist"):
        data = load_encodings()
        
        st.subheader("Add to Blacklist")
        registered_users = [(name, phone) for name, phone in zip(data["names"], data["phones"]) if name != SECURITY_NAME]
        if registered_users:
            user_options_display = [f"{name} ({phone})" for name, phone in registered_users]
            selected_index = st.selectbox(
                "Select user to blacklist", 
                range(len(registered_users)), 
                format_func=lambda x: user_options_display[x]
            )
            selected_name, selected_phone = registered_users[selected_index]
            
        if st.button("🚫 Blacklist Selected User"):
            exact_index = next((i for i, (n, p) in enumerate(zip(data["names"], data["phones"])) 
                        if n == selected_name and p == selected_phone), None)

            if exact_index is None:
                st.error("User not found in database")
                st.stop()

            # Get the image path from the exact index
            original_image_path = data["image_paths"][exact_index] if exact_index < len(data["image_paths"]) else None

            if original_image_path and os.path.exists(original_image_path):
                # Move the image to the blacklist folder
                blacklist_user_dir = BLACKLIST_IMAGES_DIR / "user"
                blacklist_user_dir.mkdir(parents=True, exist_ok=True)
                blacklist_image_path = blacklist_user_dir / Path(original_image_path).name
                os.rename(original_image_path, blacklist_image_path)
            else:
                st.warning("Image file not found, proceeding without image")
                blacklist_image_path = None

            # Add user to blacklist metadata
            data["blacklist_metadata"].append({
                "name": selected_name,
                "phone": selected_phone,
                "email": data["emails"][exact_index],
                "image_path": str(blacklist_image_path) if blacklist_image_path else None,
                "type": "user",
                "blacklisted_at": datetime.now().isoformat()
            })
            data["blacklist"].append(data["encodings"][exact_index])

            # Remove user from original lists
            for key in ["names", "phones", "emails", "encodings", "image_paths"]:
                if exact_index < len(data[key]):
                    del data[key][exact_index]

            save_encodings(data)
            update_blacklist_counts(data["blacklist_metadata"])
            st.rerun()

        st.subheader("Current Blacklist")
        if data["blacklist_metadata"]:
            blacklist_options_display = [f"{entry['name']} ({entry['phone']})" for entry in data["blacklist_metadata"]]
            selected_index = st.selectbox(
                "Select blacklisted person to remove", 
                range(len(data["blacklist_metadata"])), 
                format_func=lambda x: blacklist_options_display[x]
            )
            selected_entry = data["blacklist_metadata"][selected_index]
            sel_name = selected_entry['name']
            sel_phone = selected_entry['phone']
            
            if st.button("Remove from Blacklist"):
                meta_index = next((i for i, entry in enumerate(data["blacklist_metadata"]) 
                                if entry["name"] == sel_name and entry["phone"] == sel_phone), None)

                if meta_index is None:
                    st.error("Entry not found in blacklist")
                    st.stop()

                entry = data["blacklist_metadata"][meta_index]
                original_image_path = entry.get("image_path")

                if entry["type"] == "user":
                    # Restore user data
                    data["names"].append(entry["name"])
                    data["phones"].append(entry["phone"])
                    data["emails"].append(entry["email"])
                    data["encodings"].append(data["blacklist"][meta_index])

                    # Restore image if exists
                    if original_image_path and os.path.exists(original_image_path):
                        restore_date = datetime.now().strftime("%Y-%m-%d")
                        restore_dir = Path("user_images") / restore_date
                        restore_dir.mkdir(parents=True, exist_ok=True)
                        new_path = restore_dir / Path(original_image_path).name
                        os.rename(original_image_path, new_path)
                        data["image_paths"].append(str(new_path))

                # Remove from blacklist
                del data["blacklist_metadata"][meta_index]
                del data["blacklist"][meta_index]

                save_encodings(data)
                update_blacklist_counts(data["blacklist_metadata"])
                st.rerun()
            
            st.markdown("### Blacklisted Persons:")
            for entry in data["blacklist_metadata"]:
                st.write(f"""
                - **Name**: {entry['name']}  
                **Phone**: {entry['phone']}  
                **Type**: {entry.get('type', 'user').title()}  
                **Blacklisted**: {datetime.fromisoformat(entry.get('blacklisted_at', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M')}
                """)
        else:
            st.info("No blacklisted persons")
    with st.expander("📊 Access Logs Analytics"):
        col1, col2, col3 = st.columns(3)
            
        with col1:
            registered_users = len([n for n in data["names"] if n != SECURITY_NAME])
            st.metric("Total Registered Users", registered_users)
        
        with col2:
            total_blacklisted = len(data["blacklist_metadata"])
            st.metric("Total Blacklisted Users", total_blacklisted)
        
        with col3:
            today = datetime.now().strftime("%Y-%m-%d")
            today_logs = LOGS_DIR / today / "logs.xlsx"
            if today_logs.exists():
                df = pd.read_excel(today_logs)
                count = len(df)
            else:
                count = 0
            st.metric("Today's Entries", count)

        selected_date = st.date_input("Select date", datetime.today())
        date_path = LOGS_DIR / selected_date.strftime("%Y-%m-%d")  # Use hyphens for directory structure
        log_file = date_path / "logs.xlsx"

        if log_file.exists():
            df = pd.read_excel(log_file)

            # Convert timestamp to datetime
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            
            # Add time formatting
            df["Time"] = df["Timestamp"].dt.strftime("%H:%M:%S")
            df["Hour"] = df["Timestamp"].dt.hour
            df["Date"] = df["Timestamp"].dt.date

            # ---- New Analytics Section ----
            st.subheader("📈 Security Analytics")

            # Metrics Row 1
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Repeat visitors calculation
                visitor_counts = df['Visitor Name'].value_counts()
                repeat_visitors = (visitor_counts > 1).sum()
                st.metric("🔁 Repeat Visitors", repeat_visitors)

            with col2:
                # Average visits per visitor
                unique_visitors = df['Visitor Name'].nunique()
                avg_visits = len(df)/unique_visitors if unique_visitors > 0 else 0
                st.metric("📊 Avg Visits/Visitor", f"{avg_visits:.1f}")

            with col3:
                # Blacklist attempts
                blacklist_attempts = len(df[df['Action Type'].isin(['denied', 'blacklisted'])])
                st.metric("🚫 Blocked Attempts", blacklist_attempts)

            with col4:
                # Approval rate
                approved = len(df[df['Action Type'] == 'approved'])
                st.metric("✅ Approved Entries", approved)

            # Metrics Row 2
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                denied = len(df[df['Action Type'] == 'denied'])
                st.metric("❌ Denied Entries", denied)
            
            with col2:
                blacklisted = len(df[df['Action Type'] == 'blacklisted'])
                st.metric("⛔ Blacklisted Entries", blacklisted)
            
            with col3:
                st.metric("👥 Total Visitors", df['Visitor Name'].nunique())
            
            with col4:
                st.metric("📝 Total Entries", len(df))

            # Visitor Trends and Peak Hours
            st.subheader("🕒 Visitor Traffic Heatmap")
            try:
                # Create hourly counts dataframe
                hourly_counts = df['Hour'].value_counts().sort_index().reindex(range(24), fill_value=0)
                
                # Create heatmap data
                heatmap_data = pd.DataFrame({
                    'Hour': hourly_counts.index,
                    'Visits': hourly_counts.values
                })

                # Display as bar chart with color gradient
                st.bar_chart(heatmap_data.set_index('Hour'), use_container_width=True, 
                            color=[(0.2, 0.7, 0.3)])  # Green gradient
                
                st.caption("Darker colors indicate higher visitor traffic")
            except Exception as e:
                st.error(f"Could not generate traffic heatmap: {str(e)}")

            # Top Visitors Section
            st.subheader("🏆 Frequent Visitors")
            top_visitors = df['Visitor Name'].value_counts().head(5).reset_index()
            top_visitors.columns = ['Visitor', 'Visits']
            st.dataframe(top_visitors, hide_index=True, use_container_width=True)

            # Raw Data Table
            st.subheader("📄 Raw Log Entries")
            st.dataframe(
                df[["Action Type", "Visitor Name", "Phone Number", "Time", "Authorized By"]],
                column_config={
                    "Action Type": "Action",
                    "Visitor Name": "Name",
                    "Phone Number": "Phone",
                    "Authorized By": "Authorized By"
                },
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No logs available for selected date")
# --- Main Application Flow ---
security_gate()

st.title("Face Recognition System 👤")
if st.session_state.get('security_recognized'):
    security_dashboard()
    st.stop()
if st.button("Start Face Recognition"):
    st.session_state.capture = True
    st.session_state.unrecognized_face = None
    st.session_state.show_registration = False

if st.session_state.capture:
    img_file = st.camera_input("Position face in frame")
    if img_file:
        image = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        face_encoding = detect_and_encode(image)
        
        if face_encoding is not None:
            data = load_encodings()
            
            # Replace arcface.calc_sim with cosine_similarity
            blacklist_matches = [cosine_similarity(face_encoding, bl_enc) > 0.6 for bl_enc in data["blacklist"]]
            if any(blacklist_matches):  # Use 'any()' instead of 'True in ...'
                st.error("🚫 Access Denied - You are permanently blacklisted!")
                st.session_state.capture = False
            else:
                matches = [cosine_similarity(face_encoding, enc) > 0.6 for enc in data["encodings"]]
                
                if any(matches):
                    idx = matches.index(True)
                    name = data["names"][idx]
                    phone = data["phones"][idx]

                    if name == SECURITY_NAME:
                        st.session_state.security_recognized = True
                        st.session_state.full_auth = True  # Automatically authenticate security
                        st.success("Security recognized - Redirecting to Dashboard...")
                        st.rerun()  # Force rerun to immediately go to the dashboard
                    else:
                        log_visit(name, phone)
                        st.success(f"Welcome {name}!")
                        st.session_state.capture = False
                else:
                    st.warning("Unrecognized face - registration required")
                    st.session_state.unrecognized_face = face_encoding
                    st.session_state.show_registration = True
        else:
            st.error("No face detected")

if st.session_state.show_registration:
    with st.form("Visitor Information"):
        residents = get_registered_residents()
        resident_options = [f"{name} ({phone})" for name, phone, email in residents]
        
        visitor_name = st.text_input("Your Name")
        visitor_phone = st.text_input("Your Phone Number")
        selected_resident = st.selectbox("Who are you here to visit?", resident_options)
        
        if st.form_submit_button("Submit Request"):
            if visitor_name and visitor_phone and selected_resident:
                index = resident_options.index(selected_resident)
                _, _, resident_email = residents[index]
                image_path = save_image(image, visitor_name, visitor_phone, is_registered=False)
                st.session_state.image_path = str(image_path)  
                request_id = str(uuid.uuid4())
                if create_pending_request(visitor_name, visitor_phone, resident_email, request_id):
                    send_approval_email(visitor_name, visitor_phone, resident_email, request_id, image_path)
                    st.session_state.pending_request = request_id
                    st.session_state.show_registration = False

if st.session_state.pending_request:
    status = check_approval_status(st.session_state.pending_request)
    if status:
        if status["status"] == "approved":
    # Log the visit but DO NOT store encoding
            log_visit(status["visitor_name"], status["visitor_phone"], "approved", status["resident_email"])
            
            st.success(f"Access granted to {status['visitor_name']}!")
            st.session_state.pending_request = None
            st.session_state.capture = False
        elif status["status"] == "denied":
            st.error("Access denied by resident")
            st.session_state.pending_request = None
            st.session_state.capture = False
            if "image_path" in st.session_state:
                os.remove(st.session_state.image_path)  
                del st.session_state.image_path 
        elif status["status"] == "blacklisted":
            st.error("You have been blacklisted")
            data = load_encodings()
            visitor_image_path = save_image(image, visitor_name, visitor_phone, is_registered=False)
            
            # Add this block to organize blacklisted visitor images
            blacklist_visitor_dir = BLACKLIST_IMAGES_DIR / "visitor"
            blacklist_visitor_dir.mkdir(parents=True, exist_ok=True)
            blacklist_image_path = blacklist_visitor_dir / Path(visitor_image_path).name
            os.rename(visitor_image_path, blacklist_image_path)
            
            data["blacklist"].append(st.session_state.unrecognized_face)  # Store encoding
            data["blacklist_metadata"].append({  # Store metadata
                "name": status["visitor_name"],
                "image_path": str(blacklist_image_path),
                "type": "visitor",
                "phone": status["visitor_phone"]
            })
            save_encodings(data)
            update_blacklist_counts(data["blacklist_metadata"])
            st.session_state.pending_request = None
            st.session_state.capture = False
        else:
            st.warning("Waiting for resident approval...")
            time.sleep(3)
            st.rerun()

if st.session_state.security_recognized:
    security_dashboard()
