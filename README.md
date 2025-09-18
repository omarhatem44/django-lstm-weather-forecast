# Weather Forecasting with LSTM & Django 🌦️

This project is a **weather forecasting web application** built using **Django** for the backend and an **LSTM (Long Short-Term Memory)** model for time series prediction.  
The app allows users to enter a city name, fetch current weather data, and generate predictions using the trained deep learning model.

## 🚀 Features
- 🌍 Weather data fetching via OpenWeatherMap API.  
- 🧠 LSTM deep learning model for weather forecasting.  
- 🖥️ Django-based web application with templates.  
- 📊 Visualization of predicted vs. actual weather.  

## 🛠️ Tech Stack
- **Backend:** Django, Python  
- **Deep Learning:** TensorFlow / Keras (LSTM model)  
- **Frontend:** HTML, CSS, JavaScript (charts for visualization)  
- **API:** OpenWeatherMap  

## 📂 Project Structure
weather-forecasting-lstm-django/  
│-- manage.py  
│-- requirements.txt  
│-- .gitignore  
│-- README.md  
│  
├── weatherApp/        # Django app  
│   ├── views.py  
│   ├── models.py  
│   ├── urls.py  
│   ├── templates/  
│   └── static/  
│  
├── ml_model/          # ML related files  
│   ├── train_models.py  
│   ├── weather_lstm.h5  
│   └── preprocessing.py  
│  
└── db.sqlite3  

## ⚙️ Installation & Usage
# 1️⃣ Clone the repo
git clone https://github.com/<your-username>/weather-forecasting-lstm-django.git
cd weather-forecasting-lstm-django

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run migrations
python manage.py migrate

# 5️⃣ Start the server
python manage.py runserver

# ✅ Now open `http://127.0.0.1:8000/` in your browser 🚀

## 📦 Requirements
- Python 3.9+  
- Django  
- TensorFlow / Keras  
- Requests  
- Pandas, NumPy, Matplotlib  

(Install automatically via requirements.txt)

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 📜 License
This project is licensed under the MIT License.
