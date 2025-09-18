from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import os
from tensorflow.keras.models import load_model

API_Key = '6fc402f41ac04ab882a80aa8d7126677'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'


def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_Key}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],
        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
    }


def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def predict_future_lstm(model, data, n_steps=5, seq_length=10):
    predictions = []
    input_seq = data[-seq_length:].reshape((1, seq_length, 1))

    for _ in range(n_steps):
        next_val = model.predict(input_seq, verbose=0)  # (1,1)
        predictions.append(next_val[0, 0])
        next_val_reshaped = next_val.reshape((1, 1, 1))
        input_seq = np.concatenate([input_seq[:, 1:, :], next_val_reshaped], axis=1)

    return predictions


rain_model = load_model("models/rain_model.keras")
temp_model = load_model("models/temp_model.keras")
hum_model = load_model("models/humidity_model.keras")


def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)

        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'weather.csv')
        historical_data = pd.read_csv(csv_path).dropna().drop_duplicates()

        # ---------------- Temperature Prediction ----------------
        temps = historical_data['Temp'].values
        scaler_temp = MinMaxScaler()
        temps_scaled = scaler_temp.fit_transform(temps.reshape(-1, 1)).flatten()

        future_temps_scaled = predict_future_lstm(temp_model, temps_scaled, n_steps=5)
        future_temps = scaler_temp.inverse_transform(np.array(future_temps_scaled).reshape(-1, 1)).flatten()

        # ---------------- Humidity Prediction ----------------
        hums = historical_data['Humidity'].values
        scaler_hum = MinMaxScaler()
        hums_scaled = scaler_hum.fit_transform(hums.reshape(-1, 1)).flatten()

        future_hum_scaled = predict_future_lstm(hum_model, hums_scaled, n_steps=5)
        future_hum = scaler_hum.inverse_transform(np.array(future_hum_scaled).reshape(-1, 1)).flatten()

        # ---------------- Times ----------------
        timezone = pytz.timezone('Asia/Karachi')
        now = datetime.now(timezone)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(5)]

        # store each value separately
        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temps
        hum1, hum2, hum3, hum4, hum5 = future_hum

        context = {
            'location': city,
            'current_weather': current_weather,
            'Min_Temp': int(round(current_weather['temp_min'])),
            'Max_Temp': int(round(current_weather['temp_max'])),
            'feels_like': int(round(current_weather['feels_like'])),
            'Humidity': int(round(current_weather['humidity'])),
            'clouds': int(round(current_weather['clouds'])),
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],

            'time': datetime.now(),
            'date': datetime.now().strftime("%B, %d, %Y"),

            'wind': int(round(current_weather['Wind_Gust_Speed'])),
            'pressure': int(round(current_weather['pressure'])),
            'visibility': int(round(current_weather['visibility'])),

            'time1': time1, 'time2': time2, 'time3': time3, 'time4': time4, 'time5': time5,
            'temp1': f"{int(round(temp1))}", 'temp2': f"{int(round(temp2))}",
            'temp3': f"{int(round(temp3))}", 'temp4': f"{int(round(temp4))}",
            'temp5': f"{int(round(temp5))}",
            'hum1': f"{int(round(hum1))}", 'hum2': f"{int(round(hum2))}",
            'hum3': f"{int(round(hum3))}", 'hum4': f"{int(round(hum4))}", 'hum5': f"{int(round(hum5))}",
        }

        return render(request, 'weather.html', context)

    return render(request, 'weather.html')

