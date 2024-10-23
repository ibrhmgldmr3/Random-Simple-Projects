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