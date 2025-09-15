## Предмет: Штучний інтелект: принципи та методи

# Викладач: Марченко Олександр Олександрович — професор, гарант ОНП “Інтелектуальні системи”

# Датасет: Flight Price Prediction(https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
## 1) Постанова задачі

Необхідно побудувати систему прогнозування цін на авіаквитки.
Мета — на основі історичних даних (авіакомпанія, місто вильоту та прильоту, час відправлення, кількість днів до вильоту та інші ознаки) передбачити вартість квитка.

Вхід: категоріальні та числові ознаки (авіакомпанія, рейс, міста, час, тривалість, клас квитка, кількість днів до вильоту).

Вихід: прогноз ціни квитка (цільова змінна price).

Метрики якості: MAE, RMSE, R².

# Щоб порівняти підходи різної складності, було реалізовано три моделі:

Проста (Baseline): Лінійна регресія.

Середня: Random Forest Regressor.

Складна: XGBoost.

Усі моделі зберігають прогнози та метрики у reports/tables/, а Dashboard (Dash/Plotly) дозволяє переглядати графіки та таблицю метрик.

## 2) Опис датасету

Кількість записів: близько 300 тис.

Ознаки:

airline — авіакомпанія;

source_city — місто відправлення;

destination_city — місто призначення;

departure_time, arrival_time — часи вильоту та прильоту;

stops — кількість пересадок;

duration — тривалість у годинах;

class — клас квитка (Economy/Business);

days_left — кількість днів до вильоту;

price — цільова змінна.

# Приклад фрагменту таблиці:

airline	flight	source_city	departure_time	stops	destination_city	class	duration	days_left	price
SpiceJet	SG-8709	Delhi	Evening	zero	Mumbai	Economy	2.17	1	5953
AirAsia	I5-764	Delhi	Early_Morning	zero	Mumbai	Economy	2.17	1	5956
## 3) Структура репозиторію
<pre>
├─ dash_app/
│ ├─ assets/ # стилі Dash
│ └─ app.py # Dash-додаток
├─ data/
│ └─ flight_price.csv # датасет
├─ models/
│ ├─ baseline/ # ваги Baseline
│ ├─ rf/ # ваги RandomForest
│ └─ xgb/ # ваги XGBoost
├─ reports/
│ └─ tables/ # CSV з прогнозами та метриками
├─ src/
│ ├─ data.py # завантаження датасету
│ ├─ features.py # препроцесинг та кодування ознак
│ ├─ eval.py # MAE/RMSE/R²
│ ├─ train_baseline.py # навчання Baseline
│ ├─ train_rf.py # навчання RandomForest
  └─ train_xgb.py # навчання XGBoost
└─ </pre>

## 4) Реалізація модулів

src/data.py — завантаження flight_price.csv, підготовка ознак.

src/features.py — кодування категоріальних ознак (One-Hot Encoding).

src/eval.py — реалізація метрик MAE, RMSE, R².

train_baseline.py — лінійна регресія, простий baseline.

train_rf.py — RandomForestRegressor, ансамбль дерев.

train_xgb.py — XGBoost, градієнтний бустинг.

dash_app/app.py — Dash-додаток для візуалізації:

Overlay Actual vs Predicted.

Side-by-side для моделей.

Таблиця метрик.

## 5) Встановлення
# Віртуальне середовище
<pre> python -m venv .venv </pre>
.venv\Scripts\activate

# Залежності
<pre> pip install pandas numpy scikit-learn xgboost dash plotly </pre>


Датасет flight_price.csv покласти в data/.

## 6) Команди запуску
Baseline
<pre> python -m src.train_baseline </pre>


→ зберігає baseline_metrics.csv, baseline_forecast.csv.

Random Forest
<pre> python -m src.train_rf </pre>

XGBoost
<pre> python -m src.train_xgb </pre>

Dash-додаток
<pre> python -m dash_app.app </pre>

## 7) Хід виконання

Проведено підготовку даних: категоріальні змінні переведені у числові.

Дані розділено на 80% для навчання та 20% для тестування.

Навчені три моделі (Baseline, RF, XGBoost).

Обчислені метрики MAE, RMSE, R².

Результати збережено у CSV.

Створено Dash-додаток для наочної візуалізації.
<img width="1280" height="466" alt="image" src="https://github.com/user-attachments/assets/dec4076c-8872-4169-a4c4-c7b2130c46d0" />


## 8) Результати

Baseline показав найбільшу похибку, але продемонстрував базову залежність між ознаками та ціною.
<img width="1280" height="407" alt="image" src="https://github.com/user-attachments/assets/ff500a88-ae4d-424b-ba53-0e3edea819f6" />


RandomForest знизив похибку, добре враховує взаємодії між змінними.
<img width="1280" height="340" alt="image" src="https://github.com/user-attachments/assets/e9499881-08df-49d9-8e26-1e19488f85ce" />


XGBoost показав найкращі результати з мінімальними MAE та RMSE.

На графіках видно, що XGBoost найближче відтворює фактичні ціни квитків.

<img width="1280" height="341" alt="image" src="https://github.com/user-attachments/assets/fe88d8be-d1a6-4b8a-be80-bf0b1152cccc" />

## 9) Висновки

Нижче наведено таблицю з порівнянням метрик для трьох моделей.
Як видно зі скріншота, Baseline показує високу похибку (MAE ≈ 4230), тоді як RandomForest значно точніший (MAE ≈ 762). XGBoost демонструє проміжні результати, але теж краще за Baseline.
<img width="1280" height="156" alt="image" src="https://github.com/user-attachments/assets/835946bd-716e-4f85-93f4-7f4a1662c03e" />

Створений Dash-додаток спрощує аналіз результатів і порівняння моделей.
