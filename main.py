import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

#1 — МОДЕЛЮВАННЯ (Практикум №1)

print("ЧАСТИНА 1")

# 1. Генерація випадкової величини за нормальним законом розподілу
# Параметри: середнє (loc) = 0, стандартне відхилення (scale) = 5, кількість (size) = 100
N = 100
noise = np.random.normal(loc=0, scale=5, size=N)

# 2. Задання детермінованого тренду (лінійний тренд: y = a*x + b)
x = np.arange(N)
a = 0.5 
b = 10  
trend = a * x + b

# 3. Побудова адитивної моделі (Тренд + Стохастична складова)
sample_data = trend + noise

# 4. Розрахунок статистичних характеристик вибірки
mean_val = np.mean(sample_data)
var_val = np.var(sample_data)
std_val = np.std(sample_data)

print(f"Математичне очікування (Mean): {mean_val:.2f}")
print(f"Дисперсія (Variance): {var_val:.2f}")
print(f"Середньоквадратичне відхилення (СКВ/STD): {std_val:.2f}\n")

# 5. Побудова графіків 
plt.figure(figsize=(12, 5))

# Графік 1: Тренд та вибірка
plt.subplot(1, 2, 1)
plt.plot(x, sample_data, label='Адитивна модель (Вибірка)', marker='o', markersize=4, linestyle='-', alpha=0.7)
plt.plot(x, trend, label='Лінійний тренд', color='red', linewidth=2)
plt.title("Модель: Тренд + Випадкова складова")
plt.xlabel("Час (x)")
plt.ylabel("Значення (y)")
plt.legend()
plt.grid(True)

# Графік 2: Гістограма закону розподілу вибірки
plt.subplot(1, 2, 2)
plt.hist(sample_data, bins=15, color='skyblue', edgecolor='black')
plt.title("Гістограма розподілу вибірки")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.grid(axis='y')

plt.tight_layout()
plt.show()



# ЧАСТИНА 2 — РЕАЛЬНІ ДАНІ 

print("ЧАСТИНА 2")

# Використаю бібліотеку requests для парсингу JSON
url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=60"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    # Індекс 4 в масиві klines від Binance відповідає ціні закриття (Close price)
    close_prices = [float(day[4]) for day in data]
    
    # 2. Збереження даних у файл CSV за допомогою pandas
    df = pd.DataFrame({'Day': np.arange(1, 61), 'BTC_Close_Price': close_prices})
    csv_filename = 'real_data_btc.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Дані успішно завантажені та збережені у файл {csv_filename}")
    
    # 3. Розрахунок статистичних характеристик реальних даних
    real_mean = df['BTC_Close_Price'].mean()
    real_var = df['BTC_Close_Price'].var(ddof=0) # ddof=0 для генеральної дисперсії
    real_std = df['BTC_Close_Price'].std(ddof=0)
    
    print(f"Математичне очікування (Mean): {real_mean:.2f} USD")
    print(f"Дисперсія (Variance): {real_var:.2f}")
    print(f"Середньоквадратичне відхилення (СКВ/STD): {real_std:.2f} USD\n")
    
    # Побудова графіків 
    plt.figure(figsize=(12, 5))
    
    # Графік 1: Часовий ряд (Динаміка ціни)
    plt.subplot(1, 2, 1)
    plt.plot(df['Day'], df['BTC_Close_Price'], color='orange', marker='.', linestyle='-')
    plt.title("Динаміка ціни Bitcoin (останні 60 днів)")
    plt.xlabel("День")
    plt.ylabel("Ціна (USD)")
    plt.grid(True)
    
    # Графік 2: Гістограма реальних даних
    plt.subplot(1, 2, 2)
    plt.hist(df['BTC_Close_Price'], bins=12, color='lightgreen', edgecolor='black')
    plt.title("Гістограма розподілу цін Bitcoin")
    plt.xlabel("Ціна (USD)")
    plt.ylabel("Частота")
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Помилка при завантаженні даних з API.")

