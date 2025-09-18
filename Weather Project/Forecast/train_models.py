import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --------------------------
# تجهيز البيانات
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
# بناء الموديلات
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
# التدريب وحفظ الموديلات
# --------------------------
def train_and_save_models():
    # حمل البيانات
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'weather.csv')
    historical_data = read_historical_data(csv_path)

    os.makedirs("models", exist_ok=True)

    # -------- RainTomorrow Classification --------
    le = LabelEncoder()
    historical_data['RainTomorrow'] = le.fit_transform(historical_data['RainTomorrow'])

    rain_seq = historical_data['RainTomorrow'].values
    X_rain, y_rain = create_sequences(rain_seq, seq_length=10)
    X_rain = X_rain.reshape((X_rain.shape[0], X_rain.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_rain, y_rain, test_size=0.2, random_state=42)

    rain_model = build_lstm_classifier((X_rain.shape[1], 1))
    rain_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    rain_model.save("models/rain_model.keras")
    print("✅ RainTomorrow model saved.")

    # -------- Temperature Regression --------
    temps = historical_data['Temp'].values
    scaler_temp = MinMaxScaler()
    temps_scaled = scaler_temp.fit_transform(temps.reshape(-1, 1)).flatten()

    X_temp, y_temp = create_sequences(temps_scaled, seq_length=10)
    X_temp = X_temp.reshape((X_temp.shape[0], X_temp.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    temp_model = build_lstm_regressor((X_temp.shape[1], 1))
    temp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    temp_model.save("models/temp_model.keras")
    print("✅ Temperature model saved.")

    # -------- Humidity Regression --------
    hums = historical_data['Humidity'].values
    scaler_hum = MinMaxScaler()
    hums_scaled = scaler_hum.fit_transform(hums.reshape(-1, 1)).flatten()

    X_hum, y_hum = create_sequences(hums_scaled, seq_length=10)
    X_hum = X_hum.reshape((X_hum.shape[0], X_hum.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_hum, y_hum, test_size=0.2, random_state=42)

    hum_model = build_lstm_regressor((X_hum.shape[1], 1))
    hum_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    hum_model.save("models/humidity_model.keras")
    print("✅ Humidity model saved.")

if __name__ == "__main__":
    train_and_save_models()
