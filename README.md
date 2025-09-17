# 🌍 Carbon Sense

**Carbon Sense** is an AI-powered project that helps users track and understand their carbon emissions every time they fuel their vehicle. By scanning vehicle registration plates, fetching vehicle details, predicting emissions, and sending personalized awareness messages via SMS, Carbon Sense makes sustainability awareness fun, interactive, and impactful.

---

## 🚀 Features

* 📷 **License Plate Recognition**: Uses OpenCV, EasyOCR, and PyTesseract to detect and read vehicle registration plates at petrol pumps.
* ⛽ **Fuel Tracking**: Automatically updates MongoDB with details like liters filled, amount spent, and visit frequency.
* 🔮 **Emission Prediction**: Employs a trained **RandomForestRegressor** ML model to predict CO2 emissions based on fueling behavior and vehicle details.
* 🤖 **Smart Messaging**: Integrates with **Groq LLM** to craft funny yet serious awareness messages highlighting predicted emissions and suggesting eco-friendly steps.
* 📲 **Instant Alerts**: Sends the personalized awareness message to the user via **Twilio SMS API**.
* 🎨 **Interactive Dashboard**: Built using **Streamlit** for real-time visualization and interaction.

---

## 🛠️ Tech Stack

* **Frontend / Dashboard**: [Streamlit](https://streamlit.io/)
* **Backend**: Python, Node.js
* **Machine Learning**: RandomForestRegressor
* **Database**: MongoDB
* **Computer Vision / OCR**: OpenCV, EasyOCR, PyTesseract
* **Messaging**: Twilio SMS API
* **LLM Integration**: Groq

---

## ⚡ Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/aryangiri03/CarbonSense.git
cd carbon-sense
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file in the root directory:

```env
MONGO_URI=your_mongo_connection_string
TWILIO_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Example Workflow

1. User arrives at petrol pump 🚗
2. Registration plate is scanned 📸
3. Fuel data updated in MongoDB ⛽
4. Vehicle details fetched & emissions predicted 🔮
5. Funny but serious LLM-crafted message generated 🤖
6. Message sent via SMS 📲

---

## 🔥 Interactive Demo

You can try a **sample plate recognition & emission prediction** directly from the repo using the notebook:

```python
import cv2
import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext('sample_plate.jpg')

for detection in result:
    text = detection[1]
    print("Detected Plate:", text)

# Simulated prediction
def predict_emission(litres, visits):
    base = 200
    return base + (litres * 2.5) + (visits * 10)

print("Predicted CO2 Emission:", predict_emission(40, 5), "g/km")
```

---

## 📦 Project Structure

```
carbon-sense/
│── app.py               # Streamlit dashboard
│── carbonsense.pkl      # Trained ML model
│── models/
│   ├── PetrolPumpVisits.js # Maintains record of fuel station visits and fills.
│   ├── PucDetails.js    # Fetches the PUC status and details for vehicles.
│   ├── Vehicle.js       # Stored all details of vehicles
│   └── VehicleMetrics.js # Stores all the metrics for the vehicles
│── seed.js              # Feed sample data to database.
│── notebooks/
|    └──CarbonSense.ipynb  # Jupyter notebooks for training the model.
│── update.csv # preprocessed csv stage-1
│── preprocessed.csv # Final preprocessed csv.
│── camera.py # realtime number plate detection via camera
│── message.py # message generator through llm
│── .env   # environmental variables setup
│── README.md
```

---

## 📱 Example Awareness Message

> "Hey there! 🌍 Based on your fueling pattern, your car could emit **245g/km of CO₂** if you keep the same pace. That’s like making your car a mini volcano! 🌋 Try carpooling 🚘, biking 🚴, or even switching to an EV ⚡ for a greener tomorrow."

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and open a PR. Whether it’s improving OCR, optimizing the ML model, or enhancing the Streamlit dashboard, every contribution counts. 🙌

---

## 📜 License

This project is licensed under the MIT License.

---

## 🌟 Acknowledgements

* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [Twilio](https://www.twilio.com/)
* [Groq](https://groq.com/)

---

💡 *Carbon Sense – Because every drop of fuel counts, and so does every gram of CO₂.*

## Team:
Aayush Gattani
Aryan Giri
Harshal Bankhele
Vedant Manvelikar
