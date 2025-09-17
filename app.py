import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
from pymongo import MongoClient
import joblib
import datetime
import json
import os
import traceback
from bson.objectid import ObjectId
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import re
import easyocr

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Smart Plate Scanner 🔮", layout="wide")

# -------------------------
# Database Connection
# -------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "carbonsense")

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    petrol_col = db["petrolpumpvisits"]
    metrics_col = db["vehiclemetrics"]
    puc_col = db["pucdetails"]
    vehicles_col = db["vehicles"]
except Exception as e:
    st.error(" Could not connect to MongoDB. Check connection string or server.")
    st.exception(e)
    st.stop()

# -------------------------
# ML Model Load
# -------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "carbonsense.pkl")
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    st.warning(" Could not load ML model. Prediction will be unavailable until fixed.")
    with st.expander("Model load error"):
        st.text(str(e))
        st.text(traceback.format_exc())

# -------------------------
# LLM Setup
# -------------------------
llm = None
try:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
except Exception as e:
    st.warning(" Could not initialize LLM. Personalized messages will be unavailable.")
    with st.expander("LLM initialization error"):
        st.text(str(e))

# -------------------------
# Email Configuration
# -------------------------
EMAIL_USERNAME = os.environ.get("EMAIL_USERNAME")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    st.error(" Could not initialize EasyOCR. Please check installation.")
    st.exception(e)
    reader = None

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');
    .stApp { background: linear-gradient(180deg,#0b0b0b,#161616); color: #e6feff; font-family: 'Orbitron', sans-serif; }
    h1 { color: #00f5ff; text-shadow: 0px 0px 10px rgba(0,245,255,0.18);} 
    .neon-btn .stButton>button { background: linear-gradient(90deg,#00f5ff,#ff00f5)!important; color: white; border-radius: 12px; padding: 8px 18px; box-shadow: 0 6px 18px rgba(0,245,255,0.18); font-weight: 600; }
    .card { border-radius: 10px; padding: 18px; background: rgba(255,255,255,0.02); border: 1px solid rgba(0,245,255,0.06); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
    pre { background: #060606; padding: 12px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helper Functions
# -------------------------
def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    return processed

def extract_plate_easyocr(image):
    """Extract text from license plate image using EasyOCR."""
    if reader is None:
        return ""
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use EasyOCR to detect text
    results = reader.readtext(rgb_image, detail=0, paragraph=False)
    
    # Filter and clean results
    plate_text = ""
    for text in results:
        # Clean the text
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean_text) >= 6:  # Minimum length for a license plate
            plate_text = clean_text
            break
    
    # Format the plate text if detected
    # if plate_text:
    #     if len(plate_text) == 10:  # Common format: MH12AB1234
    #         plate_text = f"{plate_text[:2]} {plate_text[2:4]} {plate_text[4:6]} {plate_text[6:]}"
    #     elif len(plate_text) == 9:  # Common format: MH1AB1234
    #         plate_text = f"{plate_text[:2]} {plate_text[2:3]} {plate_text[3:5]} {plate_text[5:]}"
    #     elif len(plate_text) == 8:  # Common format: MHAB1234
    #         plate_text = f"{plate_text[:2]} {plate_text[2:4]} {plate_text[4:]}"
    
    return plate_text

def extract_plate(image):
    """Extract text from license plate image using multiple OCR methods."""
    # First try EasyOCR
    plate_text = extract_plate_easyocr(image)
    
    # If EasyOCR fails, fall back to Tesseract
    if not plate_text:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Try multiple Tesseract configurations
        configs = [
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        
        for config in configs:
            try:
                plate_text = pytesseract.image_to_string(processed_image, config=config).strip().upper()
                if plate_text:
                    # Clean the text
                    plate_text = re.sub(r'[^A-Z0-9 ]', '', plate_text)
                    plate_text = re.sub(r'\s+', ' ', plate_text).strip()
                    
                    # Format the plate text
                    no_space = plate_text.replace(' ', '')
                    if len(no_space) >= 6:  # Minimum length for a license plate
                        if len(no_space) == 10:  # Common format: MH12AB1234
                            plate_text = f"{no_space[:2]} {no_space[2:4]} {no_space[4:6]} {no_space[6:]}"
                        elif len(no_space) == 9:  # Common format: MH1AB1234
                            plate_text = f"{no_space[:2]} {no_space[2:3]} {no_space[3:5]} {no_space[5:]}"
                        elif len(no_space) == 8:  # Common format: MHAB1234
                            plate_text = f"{no_space[:2]} {no_space[2:4]} {no_space[4:]}"
                        break
            except:
                continue
    
    return plate_text

def _current_month_key(dt: datetime.date = None) -> str:
    dt = dt or datetime.datetime.utcnow().date()
    return dt.strftime("%Y-%m")

def _utc_now():
    return datetime.datetime.utcnow()

def get_vehicle_details(registration_number):
    """Get vehicle details from database."""
    # Clean the registration number for search
    clean_reg = re.sub(r'[^A-Z0-9]', '', registration_number.upper())
    return vehicles_col.find_one({"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}})

def get_puc_status(registration_number):
    """Get PUC status for vehicle."""
    # Clean the registration number for search
    clean_reg = re.sub(r'[^A-Z0-9]', '', registration_number.upper())
    puc = puc_col.find_one(
        {"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}},
        sort=[("validUntil", -1)]
    )
    return puc

def add_petrol_visit(registration_number, visit_data):
    """Add a petrol visit to the database."""
    # Ensure visitDate is stored as datetime
    if "visitDate" not in visit_data or not isinstance(visit_data["visitDate"], datetime.datetime):
        visit_data["visitDate"] = _utc_now()
    
    # Clean the registration number for search
    clean_reg = re.sub(r'[^A-Z0-9]', '', registration_number.upper())
    
    # Add to petrol visits collection
    result = petrol_col.update_one(
        {"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}},
        {"$push": {"visits": visit_data}, "$setOnInsert": {"registrationNumber": registration_number}},
        upsert=True
    )
    return result

def calculate_distance_driven(registration_number, current_litres):
    """Calculate distance driven based on fuel consumption patterns."""
    # Get vehicle details
    vehicle = get_vehicle_details(registration_number)
    if not vehicle:
        return 0
    
    # Get fuel efficiency (km/l)
    fuel_efficiency = vehicle.get("fuelConsumptionCombined_kmpl", 0)
    
    # If no specific efficiency data, use typical values based on vehicle type
    if fuel_efficiency <= 0:
        vehicle_class = vehicle.get("vehicleClass", "").lower()
        if "commercial" in vehicle_class:
            fuel_efficiency = 5  # km/l for commercial vehicles
        elif "suv" in vehicle_class.lower() or "four-wheeler" in vehicle_class:
            fuel_efficiency = 10  # km/l for SUVs and cars
        else:
            fuel_efficiency = 25  # km/l for smaller vehicles
    
    # Calculate distance based on fuel consumed
    distance_driven = current_litres * fuel_efficiency
    
    # Add some randomness to simulate real-world variation (10% variation)
    import random
    variation = random.uniform(0.9, 1.1)
    distance_driven *= variation
    
    return round(distance_driven, 2)

def update_vehicle_metrics(registration_number, current_litres):
    """Update vehicle metrics based on recent visits and current fuel fill."""
    # Calculate distance driven based on current fuel fill
    distance_driven = calculate_distance_driven(registration_number, current_litres)
    
    # Clean the registration number for search
    clean_reg = re.sub(r'[^A-Z0-9]', '', registration_number.upper())
    
    # Get all visits for the vehicle
    vehicle_visits = petrol_col.find_one({"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}})
    
    # Calculate total fuel for the month
    total_fuel = current_litres
    refuel_count = 1
    total_distance = distance_driven
    
    if vehicle_visits and "visits" in vehicle_visits:
        # Filter visits from the last 30 days
        cutoff_date = _utc_now() - datetime.timedelta(days=30)
        recent_visits = [v for v in vehicle_visits["visits"] if v["visitDate"] >= cutoff_date]
        
        # Add previous visits to totals
        for visit in recent_visits:
            total_fuel += visit.get("litresFilled", 0)
            refuel_count += 1
        
        # Calculate total distance for the month
        for visit in recent_visits:
            # If previous visits had distance data, use it
            if "estimatedDistance" in visit:
                total_distance += visit["estimatedDistance"]
    
    # Prepare metrics document
    month_key = _current_month_key()
    metrics_data = {
        "registrationNumber": registration_number,
        "forMonth": month_key,
        "totalFuelThisMonth": total_fuel,
        "refuelCountThisMonth": refuel_count,
        "avgRefuelLitresThisMonth": total_fuel / refuel_count,
        "totalDistanceThisMonth": total_distance,
        "avgDailyDistanceThisMonth": total_distance / 30 if total_distance > 0 else 0,
        "visitCount": refuel_count
    }
    
    # Update metrics collection
    result = metrics_col.update_one(
        {"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}, "forMonth": month_key},
        {"$set": metrics_data},
        upsert=True
    )
    
    # Also update the vehicle document with the latest distance and fuel consumption
    vehicles_col.update_one(
        {"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}},
        {"$set": {
            "Distance_Driven_Last_Month": total_distance,
            "Fuel_Consumed_Last_Month": total_fuel
        }}
    )
    
    return metrics_data, distance_driven

def prepare_model_input(registration_number):
    """Prepare input data for the ML model."""
    # Get vehicle details
    vehicle = get_vehicle_details(registration_number)
    if not vehicle:
        st.error("Vehicle not found in database.")
        return None
    
    # Clean the registration number for search
    clean_reg = re.sub(r'[^A-Z0-9]', '', registration_number.upper())
    
    # Get latest metrics
    month_key = _current_month_key()
    metrics = metrics_col.find_one(
        {"registrationNumber": {"$regex": f".*{clean_reg}.*", "$options": "i"}, "forMonth": month_key}
    )
    
    if not metrics:
        st.warning("No metrics found for this month. Using default values.")
        metrics = {
            "totalDistanceThisMonth": 0,
            "totalFuelThisMonth": 0
        }
    
    # Calculate vehicle age
    current_year = datetime.datetime.now().year
    vehicle_age = current_year - vehicle.get("yearOfManufacture", current_year)
    
    # Prepare input for model
    sample_input = {
        "Fuel Type": vehicle.get("fuelType", "Petrol"),
        "Transmission": vehicle.get("transmissions", "Manual"),
        "Vehicle_Type": vehicle.get("vehicleClass", "Four-Wheeler"),
        "Engine_Capacity": vehicle.get("engineCC", 1000) / 1000.0,  # Convert to liters
        "Cylinders": 4,  # Default value, you might want to add this to your schema
        "vehicle_age": vehicle_age,
        "Distance_Driven_Last_Month": metrics.get("totalDistanceThisMonth", 0),
        "Fuel_Consumed_Last_Month": metrics.get("totalFuelThisMonth", 0)
    }
    
    return sample_input

def get_llm_emission_message(emission_value, puc_status):
    """Get a personalized message about emissions from the LLM."""
    if not llm:
        return "LLM service not available. Please check your API configuration."
    
    puc_info = "PUC status is active." if puc_status == "Active" else "PUC status is expired and needs renewal."
    
    messages = [
        SystemMessage(content="You are an experienced writer, you will be given the number which is the carbon emissions predicted for the user for next month, if user keeps same lifestyle, you will write a funny but serious message for the user which will be sent to the user making him aware of the carbon emissions and how he can reduce it. The message should not exceed 150 words strictly. Also include information about the PUC status if it's expired. Write the message in simple english. The message is going to be sent on the email so no need to add effects for bolding or italics the font no need of ** or #."),
        HumanMessage(f"Carbon emission prediction: {emission_value} g/km. {puc_info}")
    ]
    
    try:
        result = llm.invoke(messages)
        return result.content
    except Exception as e:
        return f"Error generating message: {str(e)}"

def send_email(to_email, subject, message):
    """Send email to the vehicle owner using Gmail SMTP."""
    if not all([EMAIL_USERNAME, EMAIL_PASSWORD]):
        st.error("Email credentials not configured. Please set EMAIL_USERNAME and EMAIL_PASSWORD environment variables.")
        return False
    
    try:
        # Create message
        msg = MIMEText(message)
        msg['From'] = EMAIL_USERNAME
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Connect to Gmail SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, to_email, msg.as_string())
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# -------------------------
# Main Application
# -------------------------
st.title("🔮 Smart Plate Scanner - Carbon Sense")
st.markdown("Upload plate image or enter number plate manually to record fuel consumption and check emissions.")

# Initialize session state
if 'plate_number' not in st.session_state:
    st.session_state.plate_number = ""
if 'vehicle_details' not in st.session_state:
    st.session_state.vehicle_details = None
if 'llm_message' not in st.session_state:
    st.session_state.llm_message = ""
if 'emission_prediction' not in st.session_state:
    st.session_state.emission_prediction = None
if 'puc_status_text' not in st.session_state:
    st.session_state.puc_status_text = "Unknown"
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None
if 'plate_confirmed' not in st.session_state:
    st.session_state.plate_confirmed = False
if 'fuel_saved' not in st.session_state:
    st.session_state.fuel_saved = False

# Mode selection
mode = st.radio("Choose Input Method", ["Camera", "Manual"])

if mode == "Camera":
    st.info("Click the camera button to capture an image of the license plate. Confirm the detected plate number, then fill in the fuel details.")
    
    # Camera input
    camera_col1, camera_col2 = st.columns([2, 1])
    
    with camera_col1:
        captured_image = st.camera_input("Take a picture of the license plate")
        
        if captured_image and not st.session_state.plate_confirmed:
            # Convert the image to opencv format
            file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            try:
                plate_number = extract_plate(image)
                if plate_number:
                    st.session_state.plate_number = plate_number
                    st.session_state.camera_image = image
                    
                    st.success(f"Detected Plate: {plate_number}")
                    
                    # Show the captured image
                    st.image(image, channels="BGR", caption="Captured License Plate")
                    
                    # Confirmation buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirm Plate", use_container_width=True):
                            st.session_state.plate_confirmed = True
                            st.session_state.vehicle_details = get_vehicle_details(plate_number)
                            st.rerun()
                    with col2:
                        if st.button("🔄 Retake Image", use_container_width=True):
                            st.session_state.plate_number = ""
                            st.session_state.camera_image = None
                            st.rerun()
                else:
                    st.error("Could not detect plate number. Please try another image.")
            except Exception as e:
                st.error("Error processing image.")
                st.exception(e)
    
    # If plate is confirmed, show fuel entry form
    if st.session_state.plate_confirmed and not st.session_state.fuel_saved:
        st.subheader(f"Vehicle: {st.session_state.plate_number}")
        
        # Show vehicle details if available
        if st.session_state.vehicle_details:
            with st.expander("Vehicle Details"):
                st.json({k: v for k, v in st.session_state.vehicle_details.items() if k != '_id'})
                
                # Show fuel efficiency if available
                fuel_efficiency = st.session_state.vehicle_details.get("fuelConsumptionCombined_kmpl", 0)
                if fuel_efficiency:
                    st.info(f"Vehicle Fuel Efficiency: {fuel_efficiency} km/l")
                else:
                    st.warning("Fuel efficiency data not available. Will estimate based on vehicle type.")
        else:
            st.warning("Vehicle not found in database. Some features may not work correctly.")
        
        # Fuel entry form
        with st.form("fuel_entry_form"):
            st.markdown("### 🛢️ Fuel Entry")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                litres = st.number_input("Litres Filled", min_value=0.01, step=0.1, format="%.2f")
            with col2:
                price_per_litre = st.number_input("Price per Litre (₹)", min_value=1.0, value=100.0, step=0.1)
            with col3:
                odo = st.number_input("Odometer Reading (optional)", min_value=0.0, step=1.0, format="%.0f")
            
            col4, col5 = st.columns(2)
            with col4:
                fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            with col5:
                payment_method = st.selectbox("Payment Method", ["Cash", "Card", "UPI", "Wallet"])
            
            pump_name = st.text_input("Petrol Pump Name", "Indian Oil")
            
            submitted = st.form_submit_button("💾 Save Fuel Entry")
            
            if submitted and litres > 0:
                visit_data = {
                    "visitDate": _utc_now(),
                    "petrolPumpName": pump_name,
                    "fuelType": fuel_type,
                    "litresFilled": float(litres),
                    "pricePaid": float(litres) * float(price_per_litre),
                    "pricePerLitre": float(price_per_litre),
                    "paymentMethod": payment_method,
                    "odometerReading": float(odo) if odo else None
                }
                
                try:
                    # Add the visit to database
                    add_petrol_visit(st.session_state.plate_number, visit_data)
                    
                    # Update metrics and calculate distance
                    metrics, distance_driven = update_vehicle_metrics(st.session_state.plate_number, float(litres))
                    
                    # Add estimated distance to the visit data
                    petrol_col.update_one(
                        {"registrationNumber": {"$regex": f".*{re.sub(r'[^A-Z0-9]', '', st.session_state.plate_number.upper())}.*", "$options": "i"}, 
                         "visits.visitDate": visit_data["visitDate"]},
                        {"$set": {"visits.$.estimatedDistance": distance_driven}}
                    )
                    
                    st.success("✅ Fuel entry saved successfully!")
                    st.session_state.fuel_saved = True
                    
                    if metrics:
                        with st.expander("View Updated Metrics"):
                            st.json(metrics)
                            
                        # Show estimated distance
                        st.info(f"Estimated distance from this refuel: {distance_driven:.2f} km")
                        st.info(f"Total distance this month: {metrics.get('totalDistanceThisMonth', 0):.2f} km")
                except Exception as e:
                    st.error("Error saving fuel entry.")
                    st.exception(e)
    
    # After fuel is saved, automate the prediction and email sending
    if st.session_state.fuel_saved:
        with st.spinner("Analyzing data and generating report..."):
            # Prepare input for model
            model_input = prepare_model_input(st.session_state.plate_number)
            
            if model_input:
                # Get PUC status
                puc_status = get_puc_status(st.session_state.plate_number)
                puc_status_text = "Unknown"
                
                if puc_status:
                    puc_status_text = puc_status.get("status", "Unknown")
                    valid_until = puc_status.get("validUntil", "N/A")
                    
                    # Check if PUC is expired
                    if puc_status_text == "Active" and valid_until != "N/A":
                        if isinstance(valid_until, datetime.datetime) and valid_until < datetime.datetime.now():
                            puc_status_text = "Expired"
                    
                    # Store PUC status in session state
                    st.session_state.puc_status_text = puc_status_text
                
                # Make prediction if model is available
                if model:
                    try:
                        # Convert to DataFrame for model prediction
                        input_df = pd.DataFrame([model_input])
                        prediction = model.predict(input_df)[0]
                        st.session_state.emission_prediction = prediction
                        
                        # Get LLM message about emissions
                        llm_message = get_llm_emission_message(prediction, puc_status_text)
                        st.session_state.llm_message = llm_message
                        
                    except Exception as e:
                        st.error("Error making prediction.")
                        st.exception(e)
                
                # Send email if vehicle details available
                if st.session_state.vehicle_details:
                    owner_email = st.session_state.vehicle_details.get("owneremail")
                    if owner_email and st.session_state.llm_message:
                        subject = f"Carbon Emission Report for Vehicle {st.session_state.plate_number}"
                        
                        # Create a more detailed email message
                        email_message = f"""
Dear Vehicle Owner {st.session_state.vehicle_details.get('ownername', 'User')},

Here is your carbon emission report for vehicle {st.session_state.plate_number}:

Emission Prediction: {st.session_state.emission_prediction:.2f} g/km
PUC Status: {st.session_state.puc_status_text}

{st.session_state.llm_message}

Vehicle Details:
- Make: {st.session_state.vehicle_details.get('make', 'N/A')}
- Model: {st.session_state.vehicle_details.get('model', 'N/A')}
- Fuel Type: {st.session_state.vehicle_details.get('fuelType', 'N/A')}

Thank you for using Carbon Sense to monitor your vehicle's environmental impact.

Best regards,
Carbon Sense Team
"""
                        if send_email(owner_email, subject, email_message):
                            st.success("✅ Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email. Please check your email configuration.")
                
                # Show results
                st.subheader("Results")
                if st.session_state.emission_prediction:
                    st.metric("Estimated CO₂ Emission", f"{st.session_state.emission_prediction:.2f} g/km")
                st.info(f"PUC Status: {st.session_state.puc_status_text}")
                
                st.subheader("🌍 Environmental Impact Message")
                st.info(st.session_state.llm_message)
                
                # Show the captured image
                st.image(st.session_state.camera_image, channels="BGR", caption="Captured License Plate")
                
                # Reset button
                if st.button("🔄 Start New Process"):
                    st.session_state.plate_confirmed = False
                    st.session_state.fuel_saved = False
                    st.session_state.plate_number = ""
                    st.session_state.camera_image = None
                    st.rerun()

else:  # Manual mode
    plate_input = st.text_input("Enter Vehicle Registration Number", value=st.session_state.plate_number)
    if plate_input:
        st.session_state.plate_number = plate_input.upper().strip()
        st.session_state.vehicle_details = get_vehicle_details(st.session_state.plate_number)

    # If we have a plate number, show the forms (manual mode only)
    if st.session_state.plate_number:
        st.subheader(f"Vehicle: {st.session_state.plate_number}")
        
        # Show vehicle details if available
        if st.session_state.vehicle_details:
            with st.expander("Vehicle Details"):
                st.json({k: v for k, v in st.session_state.vehicle_details.items() if k != '_id'})
                
                # Show fuel efficiency if available
                fuel_efficiency = st.session_state.vehicle_details.get("fuelConsumptionCombined_kmpl", 0)
                if fuel_efficiency:
                    st.info(f"Vehicle Fuel Efficiency: {fuel_efficiency} km/l")
                else:
                    st.warning("Fuel efficiency data not available. Will estimate based on vehicle type.")
        else:
            st.warning("Vehicle not found in database. Some features may not work correctly.")
        
        # Fuel entry form
        with st.form("fuel_entry_form"):
            st.markdown("### 🛢️ Fuel Entry")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                litres = st.number_input("Litres Filled", min_value=0.01, step=0.1, format="%.2f")
            with col2:
                price_per_litre = st.number_input("Price per Litre (₹)", min_value=1.0, value=100.0, step=0.1)
            with col3:
                odo = st.number_input("Odometer Reading (optional)", min_value=0.0, step=1.0, format="%.0f")
            
            col4, col5 = st.columns(2)
            with col4:
                fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            with col5:
                payment_method = st.selectbox("Payment Method", ["Cash", "Card", "UPI", "Wallet"])
            
            pump_name = st.text_input("Petrol Pump Name", "Indian Oil")
            
            submitted = st.form_submit_button("💾 Save Fuel Entry")
            
            if submitted and litres > 0:
                visit_data = {
                    "visitDate": _utc_now(),
                    "petrolPumpName": pump_name,
                    "fuelType": fuel_type,
                    "litresFilled": float(litres),
                    "pricePaid": float(litres) * float(price_per_litre),
                    "pricePerLitre": float(price_per_litre),
                    "paymentMethod": payment_method,
                    "odometerReading": float(odo) if odo else None
                }
                
                try:
                    # Add the visit to database
                    add_petrol_visit(st.session_state.plate_number, visit_data)
                    
                    # Update metrics and calculate distance
                    metrics, distance_driven = update_vehicle_metrics(st.session_state.plate_number, float(litres))
                    
                    # Add estimated distance to the visit data
                    petrol_col.update_one(
                        {"registrationNumber": {"$regex": f".*{re.sub(r'[^A-Z0-9]', '', st.session_state.plate_number.upper())}.*", "$options": "i"}, 
                         "visits.visitDate": visit_data["visitDate"]},
                        {"$set": {"visits.$.estimatedDistance": distance_driven}}
                    )
                    
                    st.success("✅ Fuel entry saved successfully!")
                    
                    if metrics:
                        with st.expander("View Updated Metrics"):
                            st.json(metrics)
                            
                        # Show estimated distance
                        st.info(f"Estimated distance from this refuel: {distance_driven:.2f} km")
                        st.info(f"Total distance this month: {metrics.get('totalDistanceThisMonth', 0):.2f} km")
                except Exception as e:
                    st.error("Error saving fuel entry.")
                    st.exception(e)
        
        # Emissions checking section
        st.markdown("---")
        st.markdown("### 🌡️ Emissions Check")
        
        if st.button("Check Emissions and PUC Status"):
            with st.spinner("Analyzing data..."):
                # Prepare input for model
                model_input = prepare_model_input(st.session_state.plate_number)
                
                if model_input:
                    st.subheader("Model Input")
                    st.json(model_input)
                    
                    # Get PUC status
                    puc_status = get_puc_status(st.session_state.plate_number)
                    puc_status_text = "Unknown"
                    
                    if puc_status:
                        puc_status_text = puc_status.get("status", "Unknown")
                        valid_until = puc_status.get("validUntil", "N/A")
                        
                        # Check if PUC is expired
                        if puc_status_text == "Active" and valid_until != "N/A":
                            if isinstance(valid_until, datetime.datetime) and valid_until < datetime.datetime.now():
                                puc_status_text = "Expired"
                        
                        st.subheader("PUC Status")
                        st.json({
                            "Certificate Number": puc_status.get("certificateNumber", "N/A"),
                            "Issue Date": puc_status.get("issueDate", "N/A"),
                            "Valid Until": puc_status.get("validUntil", "N/A"),
                            "Status": puc_status_text
                        })
                        
                        if puc_status_text == "Expired":
                            st.error("⚠️ Your PUC certificate has expired! Please renew it immediately.")
                        elif puc_status_text == "Active":
                            st.success("✅ Your PUC certificate is active.")
                    else:
                        st.warning("No PUC certificate found for this vehicle.")
                        puc_status_text = "Not Found"
                    
                    # Store PUC status in session state
                    st.session_state.puc_status_text = puc_status_text
                    
                    # Make prediction if model is available
                    if model:
                        try:
                            # Convert to DataFrame for model prediction
                            input_df = pd.DataFrame([model_input])
                            prediction = model.predict(input_df)[0]
                            st.session_state.emission_prediction = prediction
                            
                            st.subheader("Emission Prediction")
                            st.metric("Estimated CO₂ Emission", f"{prediction:.2f} g/km")
                            
                            # Get LLM message about emissions
                            llm_message = get_llm_emission_message(prediction, puc_status_text)
                            st.session_state.llm_message = llm_message
                            
                            st.subheader("🌍 Environmental Impact Message")
                            st.info(llm_message)
                            
                            # Provide interpretation
                            if prediction < 100:
                                st.success("✅ Low emissions - Your vehicle is environmentally friendly!")
                            elif prediction < 150:
                                st.info("ℹ️ Moderate emissions - Consider regular maintenance for better efficiency.")
                            else:
                                st.warning("⚠️ High emissions - Please check your vehicle's condition and PUC status.")
                        except Exception as e:
                            st.error("Error making prediction.")
                            st.exception(e)
                    else:
                        st.warning("ML model not available. Cannot make emission prediction.")
        
        # Email sending section
        if st.session_state.llm_message and st.session_state.vehicle_details:
            st.markdown("---")
            st.markdown("### 📧 Send Report to Owner")
            
            owner_email = st.session_state.vehicle_details.get("owneremail")
            if owner_email:
                st.info(f"Owner email: {owner_email}")
                
                if st.button("Send Email Report"):
                    with st.spinner("Sending email..."):
                        subject = f"Carbon Emission Report for Vehicle {st.session_state.plate_number}"
                        
                        # Create a more detailed email message
                        email_message = f"""
    Dear Vehicle Owner,

    Here is your carbon emission report for vehicle {st.session_state.plate_number}:

    Emission Prediction: {st.session_state.emission_prediction:.2f} g/km
    PUC Status: {st.session_state.puc_status_text}

    {st.session_state.llm_message}

    Vehicle Details:
    - Make: {st.session_state.vehicle_details.get('make', 'N/A')}
    - Model: {st.session_state.vehicle_details.get('model', 'N/A')}
    - Fuel Type: {st.session_state.vehicle_details.get('fuelType', 'N/A')}

    Thank you for using Carbon Sense to monitor your vehicle's environmental impact.

    Best regards,
    Carbon Sense Team
    """
                        
                        if send_email(owner_email, subject, email_message):
                            st.success("✅ Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email. Please check your email configuration.")
            else:
                st.warning("No email address found for this vehicle owner.")

# Footer with debug info
st.markdown("---")
with st.expander("Developer Info"):
    st.write("MongoDB URI:", MONGO_URI)
    st.write("Database:", DB_NAME)
    st.write("Model loaded:", model is not None)
    st.write("LLM loaded:", llm is not None)
    st.write("Email configured:", all([EMAIL_USERNAME, EMAIL_PASSWORD]))
    st.write("EasyOCR loaded:", reader is not None)
    
    try:
        st.write("Collection counts:")
        st.write(f"- Petrol Visits: {petrol_col.count_documents({})}")
        st.write(f"- Vehicle Metrics: {metrics_col.count_documents({})}")
        st.write(f"- PUC Details: {puc_col.count_documents({})}")
        st.write(f"- Vehicles: {vehicles_col.count_documents({})}")
    except Exception as e:
        st.write("Error fetching counts:", str(e))
