import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime, timedelta
import pytz
import os

API_Key = '6fc402f41ac04ab882a80aa8d7126677'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'


# --------------------------
# جلب الطقس الحالي من API
# --------------------------
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
        'Wind_Gust_Speed': data['wind']['speed']
    }


# --------------------------
# تجهيز البيانات من CSV
# --------------------------
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


# --------------------------
# موديلات LSTM
# --------------------------
def build_lstm_regressor(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def build_lstm_classifier(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --------------------------
# التنبؤ بالقيم المستقبلية (مع إعادة تشكيل للأبعاد)
# --------------------------
def predict_future_lstm(model, data, n_steps=5, seq_length=10):
    predictions = []
    input_seq = data[-seq_length:].reshape((1, seq_length, 1))

    for _ in range(n_steps):
        next_val = model.predict(input_seq, verbose=0)  # (1,1)
        predictions.append(next_val[0, 0])
        next_val_reshaped = next_val.reshape((1, 1, 1))
        input_seq = np.concatenate([input_seq[:, 1:, :], next_val_reshaped], axis=1)

    return predictions


# --------------------------
# البرنامج الرئيسي
# --------------------------
def weather_view():
    city = input("Enter any city name :  ")
    current_weather = get_current_weather(city)

    # Load historical data
    historical_data = read_historical_data(r'D:\DL\project after certificate\Weather Forecasting\Data\weather.csv')

    # ---------------- RainTomorrow Classification ----------------
    le = LabelEncoder()
    historical_data['RainTomorrow'] = le.fit_transform(historical_data['RainTomorrow'])

    rain_seq = historical_data['RainTomorrow'].values
    X_rain, y_rain = create_sequences(rain_seq, seq_length=10)
    X_rain = X_rain.reshape((X_rain.shape[0], X_rain.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_rain, y_rain, test_size=0.2, random_state=42)

    rain_model = build_lstm_classifier((X_rain.shape[1], 1))
    rain_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    rain_model.save("rain_model.keras")

    rain_pred = rain_model.predict(X_test)
    rain_pred_class = (rain_pred > 0.5).astype(int)

    # ---------------- Temperature Regression ----------------
    temps = historical_data['Temp'].values
    scaler_temp = MinMaxScaler()
    temps_scaled = scaler_temp.fit_transform(temps.reshape(-1, 1)).flatten()

    X_temp, y_temp = create_sequences(temps_scaled, seq_length=10)
    X_temp = X_temp.reshape((X_temp.shape[0], X_temp.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    temp_model = build_lstm_regressor((X_temp.shape[1], 1))
    temp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    temp_model.save("temp_model.keras")

    future_temps_scaled = predict_future_lstm(temp_model, temps_scaled, n_steps=5)
    future_temps = scaler_temp.inverse_transform(np.array(future_temps_scaled).reshape(-1, 1)).flatten()

    # ---------------- Humidity Regression ----------------
    hums = historical_data['Humidity'].values
    scaler_hum = MinMaxScaler()
    hums_scaled = scaler_hum.fit_transform(hums.reshape(-1, 1)).flatten()

    X_hum, y_hum = create_sequences(hums_scaled, seq_length=10)
    X_hum = X_hum.reshape((X_hum.shape[0], X_hum.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_hum, y_hum, test_size=0.2, random_state=42)

    hum_model = build_lstm_regressor((X_hum.shape[1], 1))
    hum_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    hum_model.save("humidity_model.keras")

    future_hum_scaled = predict_future_lstm(hum_model, hums_scaled, n_steps=5)
    future_hum = scaler_hum.inverse_transform(np.array(future_hum_scaled).reshape(-1, 1)).flatten()

    # ---------------- Display Results ----------------
    timezone = pytz.timezone('Asia/Karachi')
    now = datetime.now(timezone)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(5)]

    print(f"\nCurrent weather in {current_weather['city']}, {current_weather['country']}:")
    print(f"Temperature: {current_weather['current_temp']}°C (Feels like {current_weather['feels_like']}°C)")
    print(f"Min Temperature: {current_weather['temp_min']}°C")
    print(f"Max Temperature: {current_weather['temp_max']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Pressure: {current_weather['pressure']} hPa")
    print(f"Wind: {current_weather['Wind_Gust_Speed']} m/s, Direction: {current_weather['wind_gust_dir']}°")
    print(f"Description: {current_weather['description']}")

    print("\nRain Prediction for Tomorrow (based on LSTM):", "Yes" if rain_pred_class[0][0] == 1 else "No")

    print("\nFuture Temperature Predictions for next 5 hours (LSTM):")
    for time, temp in zip(future_times, future_temps):
        print(f"{time}: {temp:.2f}°C")

    print("\nFuture Humidity Predictions for next 5 hours (LSTM):")
    for time, hum in zip(future_times, future_hum):
        print(f"{time}: {hum:.2f}%")


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    weather_view()
