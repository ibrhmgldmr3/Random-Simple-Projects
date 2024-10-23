# Description: Borsa verileri üzerinde makine öğrenimi modelleri oluşturmak için örnekler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. CSV dosyasını okuma
df = pd.read_csv('bist100 copy.csv', sep=';', parse_dates=['Date (DD.MM.YYYY)'])

# 2. Veriyi inceleme
print(df.head())

# 3. Tarih ve diğer kategorik sütunları işleme (gerekirse)
df['Year'] = df['Date (DD.MM.YYYY)'].dt.year
df['Month'] = df['Date (DD.MM.YYYY)'].dt.month
df['Day'] = df['Date (DD.MM.YYYY)'].dt.day

# 4. Kullanılmayan sütunları çıkarma ve NaN verileri temizleme
df_cleaned = df.drop(['Date (DD.MM.YYYY)', 'Index Code', 'Index Names In English'], axis=1)
df_cleaned = df_cleaned.dropna()

# 5. Özellikler (features) ve hedef değişken (target) belirleme
X = df_cleaned[['Year', 'Month', 'Day', 'Number of Days']]  # Özellikler
y = df_cleaned['Value']  # Tahmin etmek istediğimiz değer (hedef değişken)

# 6. Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Modeli test etme ve sonuçları değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 9. Sonuçları görselleştirme
plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen')
plt.show()

#################################################### ARIMA ####################################################
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

# Veriyi okuma
df = pd.read_csv('bist100.csv', sep=';', parse_dates=['Date (DD.MM.YYYY)'], dayfirst=True)
df.columns = df.columns.str.strip()  # Sütun adlarındaki boşlukları temizleme

# Tarihi indeks olarak ayarlama
df.set_index('Date (DD.MM.YYYY)', inplace=True)

# Yalnızca "Value" sütununa odaklanma (zaman serisi)
value_series = df['Value']

# 7 gün sonrasını tahmin etmek için ARIMA modeli eğitme
model = ARIMA(value_series, order=(5, 1, 0))  # ARIMA(p,d,q), burada p=5, d=1, q=0
model_fit = model.fit()

# Tahmin yapma (7 gün sonrası için)
forecast = model_fit.forecast(steps=7)
print(forecast)

# Tahmin edilen sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(value_series.index, value_series, label="Gerçek Değerler")
plt.plot(pd.date_range(value_series.index[-1], periods=8, freq='D')[1:], forecast, label="Tahmin Edilen Değerler", color='red')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.title('Gelecek 7 Gün İçin Tahmin')
plt.legend()
plt.show()

#################################################### Neural Network ####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Veriyi okuma
df = pd.read_csv('bist100.csv', sep=';', parse_dates=['Date (DD.MM.YYYY)'], dayfirst=True)
df.columns = df.columns.str.strip()  # Sütun adlarını temizleme
df.set_index('Date (DD.MM.YYYY)', inplace=True)

# 2. "Value" sütununu seçme ve normalleştirme
value_series = df['Value'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))  # Veriyi 0-1 aralığında ölçekleyin
value_scaled = scaler.fit_transform(value_series)

# 3. Veriyi eğitim için hazırlama (Zaman serisi şeklinde giriş çıkışları hazırlıyoruz)
def create_dataset(data, time_step=500):
    X, y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:(i+time_step), 0])  # Time step kadar geçmiş veri alıyoruz
        y.append(data[i + time_step, 0])    # Tahmin etmek istediğimiz günü ekliyoruz
    return np.array(X), np.array(y)

time_step = 500  # Son 60 güne bakarak tahmin yapacağız
X, y = create_dataset(value_scaled, time_step)

# 4. Girdi verisinin şekillendirilmesi (LSTM için [samples, time steps, features] şeklinde olması gerekir)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 5. Eğitim (%70), doğrulama (%10) ve test (%20) setlerine ayırma
train_size = int(len(X) * 0.7)
validation_size = int(len(X) * 0.1)
test_size = len(X) - train_size - validation_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + validation_size], X[train_size + validation_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + validation_size], y[train_size + validation_size:]

# 6. LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# 7. Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# 8. Erken durdurma callback'i
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 9. Modeli eğitme (doğrulama seti ile)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=3, callbacks=[early_stop])

# 10. Modeli test etme ve tahmin yapma
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 11. Veriyi geri ölçekleme
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# 12. Tüm veri üzerinde gelecek haftayı tahmin etme
x_input = value_scaled[-time_step:].reshape(1, -1)  # Son 60 günü giriş olarak al
temp_input = list(x_input[0])
lst_output = []

n_steps = time_step
for i in range(7):  # Gelecek 7 gün tahmin etme
    x_input = np.array(temp_input[-n_steps:]).reshape((1, n_steps, 1))
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    lst_output.append(yhat[0][0])

# 13. Tahmin edilen veriyi geri ölçekleme
lst_output_scaled = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# 14. Tahmin edilen 7 günlük veriyi görselleştirme
plt.plot(np.arange(1, time_step+1), scaler.inverse_transform(value_scaled[-time_step:]), label="Geçmiş 60 Gün")
plt.plot(np.arange(time_step+1, time_step+8), lst_output_scaled, label="Tahmin Edilen 7 Gün", color='red')
plt.xlabel('Günler')
plt.ylabel('Değer')
plt.legend()
plt.show()
