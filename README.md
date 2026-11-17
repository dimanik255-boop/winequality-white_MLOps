# Wine Quality ML Service

Мини-проект: от EDA белого вина (UCI Wine Quality) до API и UI.

**Задача:** регрессия — предсказать экспертную оценку `quality` (примерно 3–9) по 11 физико-химическим признакам.

**Модель:** `RandomForestRegressor` (по результатам EDA и сравнения с Ridge / DecisionTree дубликаты **оставлены** — с ними MAE лучше).

## Признаки

| Признак | Смысл |
|---------|--------|
| fixed acidity | Фиксированная кислотность, г/дм³ |
| volatile acidity | Летучая кислотность, г/дм³ |
| citric acid | Лимонная кислота, г/дм³ |
| residual sugar | Остаточный сахар, г/дм³ |
| chlorides | Хлориды, г/дм³ |
| free sulfur dioxide | Свободный SO₂, мг/дм³ |
| total sulfur dioxide | Общий SO₂, мг/дм³ |
| density | Плотность, г/см³ |
| pH | pH |
| sulphates | Сульфаты, г/дм³ |
| alcohol | Алкоголь, % об. (самый сильный признак по EDA) |

## Структура

```text
wine_quality_mlops/
├── data/           # train.csv, test.csv (после prepare)
├── src/
│   ├── prepare.py  # загрузка UCI + split
│   └── train.py    # обучение RF, metrics, model.joblib
├── models/         # model.joblib
├── reports/        # metrics_report.txt
├── api/app.py      # FastAPI: /health, /predict
├── ui/app.py       # Streamlit-форма
├── tests/          # pytest для API
├── params.yaml
├── dvc.yaml
└── README.md
```

## Установка

```bash
cd wine_quality_mlops
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# или: pip install -e ".[dev]"  при работе через pyproject.toml
```

Нужны: `pandas`, `scikit-learn`, `joblib`, `PyYAML`, `fastapi`, `uvicorn`, `streamlit`, `requests`, `pytest`.

## Подготовка данных и обучение

```bash
# 1) Скачать данные с UCI и сделать train/test
python -m src.prepare

# 2) Обучить модель и сохранить models/model.joblib
python -m src.train
```

Либо через DVC (если установлен):

```bash
dvc repro
```

Ожидаемые метрики на test (ориентир, зависит от разбиения): MAE ≈ 0.45–0.55, R² ≈ 0.40–0.50.

Параметры в `params.yaml` (`model.n_estimators`, `max_depth`, `data.drop_duplicates`).

## Запуск API

```bash
uvicorn api.app:app --reload --port 8000
```

- `GET http://localhost:8000/health` → `{"model_ready": true}`
- `POST http://localhost:8000/predict`

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.001,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8
  }'
```

Пример ответа:

```json
{
  "predicted_quality": 5.72,
  "unit": "expert score (approx. 3–9)",
  "interpretation": "Среднее качество"
}
```

## Запуск UI

В другом терминале:

```bash
streamlit run ui/app.py
```

Откроется форма; UI ходит в API (`API_URL`, по умолчанию `http://localhost:8000`).

## Docker Compose (опционально)

```bash
docker compose up --build
```

UI: http://localhost:8501 · API: http://localhost:8000  
Внутри сети UI использует `http://api:8000`.

## Тесты

```bash
pytest -q tests
```

## Качество модели (после обучения)

Смотри `metrics.json` и `reports/metrics_report.txt`.  
В EDA лучший вариант: **RandomForest с дубликатами**, Test MAE ≈ **0.50**.

## Замечания из EDA

- Пропусков нет; ~19% полных дубликатов — по сравнению моделей их лучше **не удалять**.
- `quality` несбалансирован (много 5–6).
- Кодирование не нужно (все признаки числовые).
- Масштабирование для RandomForest не обязательно.
