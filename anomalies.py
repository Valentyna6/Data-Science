import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


#ВИЯВЛЕННЯ ТА ПРИГНІЧЕННЯ АНОМАЛІЙ 

print("ОБРОБКА АНОМАЛІЙ")

# 1. Генерація модельних даних 
np.random.seed(42) 
N = 100
x = np.arange(N)
trend = 0.5 * x + 10
noise = np.random.normal(loc=0, scale=5, size=N)
data_original = trend + noise

# 2. Штучне додавання аномалій (викидів)
data_with_anomalies = data_original.copy()
data_with_anomalies[20] += 60  
data_with_anomalies[70] -= 50 

# 3. Розрахунок статистики ДО обробки
mean_before = np.mean(data_with_anomalies)
std_before = np.std(data_with_anomalies)

print("Статистика ДО обробки аномалій:")
print(f"Математичне очікування: {mean_before:.2f}")
print(f"СКВ (Стандартне відхилення): {std_before:.2f}\n")

# 4. Виявлення аномалій на основі Z-score
# Порогове значення = 2.5 (все, що відхиляється від середнього більше ніж на 2.5 сигми, є аномалією)
threshold = 2.5
z_scores = np.abs((data_with_anomalies - mean_before) / std_before)
anomaly_indices = np.where(z_scores > threshold)[0]

print(f"Виявлено аномалії на індексах: {anomaly_indices}")

# 5. Заміна виявлених аномалій (замінюємо на медіану вибірки)
data_cleaned = data_with_anomalies.copy()
median_val = np.median(data_with_anomalies)
for idx in anomaly_indices:
    data_cleaned[idx] = median_val

# 6. Розрахунок статистики ПІСЛЯ обробки
mean_after = np.mean(data_cleaned)
std_after = np.std(data_cleaned)

print("Статистика ПІСЛЯ обробки аномалій:")
print(f"Математичне очікування: {mean_after:.2f}")
print(f"СКВ (Стандартне відхилення): {std_after:.2f}\n")

# 7. Побудова графіків (До і Після)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, data_with_anomalies, label='Дані з аномаліями', color='red', marker='o', markersize=4)
plt.title("Дані ДО обробки (з аномаліями)")
plt.xlabel("Індекс")
plt.ylabel("Значення")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, data_cleaned, label='Очищені дані', color='blue', marker='o', markersize=4)
plt.title("Дані ПІСЛЯ заміни аномалій на медіану")
plt.xlabel("Індекс")
plt.ylabel("Значення")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ДЕКОМПОЗИЦІЯ

print("ДЕКОМПОЗИЦІЯ РЕАЛЬНИХ ДАНИХ")

# 1. Завантаження реальних даних з main.py
csv_filename = 'real_data_btc.csv'
try:
    df = pd.read_csv(csv_filename)
    
    # Використовуємо ціну закриття для аналізу
    time_series = df['BTC_Close_Price']
    
    # 2. Декомпозиція часового ряду
    decomposition = seasonal_decompose(time_series, model='additive', period=7)
    
    trend_component = decomposition.trend
    seasonal_component = decomposition.seasonal
    residual_component = decomposition.resid
    
    # 3. Візуалізація складників декомпозиції
    plt.figure(figsize=(10, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_series, label='Оригінальний ряд', color='black')
    plt.legend(loc='upper left')
    plt.title('Адитивна декомпозиція часового ряду (Bitcoin)')
    plt.ylabel('Ціна')
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(trend_component, label='Тренд (Trend)', color='blue')
    plt.legend(loc='upper left')
    plt.ylabel('Тренд')
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(seasonal_component, label='Сезонність (Seasonality)', color='green')
    plt.legend(loc='upper left')
    plt.ylabel('Сезонність')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(residual_component, label='Залишок (Residuals/Noise)', color='red', marker='o', linestyle='none', markersize=3)
    plt.legend(loc='upper left')
    plt.ylabel('Залишок')
    plt.xlabel('Дні')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Декомпозиція успішно виконана. Графіки побудовано.")

except FileNotFoundError:
    print(f"Помилка: Файл {csv_filename} не знайдено! Переконайтеся, що ви запустили код з Практикуму №2 у цій же папці.")
