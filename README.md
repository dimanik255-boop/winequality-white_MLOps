# Wine Quality ML Service

Проект по машинному обучению, от исследования данных белого вина до полноценного ML-сервиса с API и пользовательским интерфейсом

## Задача

Регрессия, предсказать экспертную оценку качества белого вина `quality` (примерная шкала 3–9) по 11 физико-химическим характеристикам

**Датасет:** UCI Wine Quality - White Wine

**Модель** `RandomForestRegressor`

По результатам EDA и сравнения с Ridge и DecisionTree выбран RandomForest. дубликаты оставлены, потому что в проведённых экспериментах модель с дубликатами показала лучший результат

## Используемые признаки

| Признак | Описание |
|---|---|
| `fixed acidity` | Фиксированная кислотность, г/дм³ |
| `volatile acidity` | Летучая кислотность, г/дм³ |
| `citric acid` | Лимонная кислота, г/дм³ |
| `residual sugar` | Остаточный сахар, г/дм³ |
| `chlorides` | Хлориды, г/дм³ |
| `free sulfur dioxide` | Свободный SO₂, мг/дм³ |
| `total sulfur dioxide` | Общий SO₂, мг/дм³ |
| `density` | Плотность, г/см³ |
| `pH` | Уровень pH |
| `sulphates` | Сульфаты, г/дм³ |
| `alcohol` | Содержание алкоголя, % об. |

По результатам EDA наиболее значимым признаком оказался `alcohol`

## Структура проекта

```text
wine_quality_mlops/
├── data/                 # train.csv, test.csv (после prepare)
├── src/
│   ├── prepare.py        # загрузка UCI + split
│   └── train.py          # обучение RF, метрики, model.joblib
├── models/               # model.joblib
├── reports/              # metrics_report.txt
├── api/
│   └── app.py            # FastAPI: /health, /predict
├── ui/
│   └── app.py            # Streamlit-форма
├── tests/
│   └── test_api.py       # тесты API
├── params.yaml           # параметры экспериментов
├── dvc.yaml              # ML pipeline
├── dvc.lock              # зафиксированные зависимости pipeline
├── metrics.json          # метрики модели
├── Dockerfile.api
├── Dockerfile.ui
├── compose.yaml
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

## Установка

Проект рассчитан на Python 3.11.

Для Windows PowerShell

```powershell
cd C:\Users\diman\Downloads\wine_quality_mlops

py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Основные библиотеки

- pandas и numpy - обработка данных
- scikit-learn - обучение и оценка модели
- joblib - сохранение модели
- PyYAML - чтение параметров
- FastAPI и uvicorn - API
- Streamlit - пользовательский интерфейс
- requests - HTTP-запросы из UI
- pytest - автоматические тесты

## Подготовка данных и обучение

Сначала загружаются данные UCI Wine Quality White, выполняется разделение на обучающую и тестовую выборки, после чего CSV-файлы сохраняются в `data/`

```powershell
python -m src.prepare
```

Затем запускается обучение

```powershell
python -m src.train
```

После обучения создаются

- `models/model.joblib` - обученная модель и информация о признаках
- `metrics.json` - метрики качества
- `reports/metrics_report.txt` - текстовый отчёт с метриками и важностью признаков

Параметры подготовки и модели находятся в `params.yaml`

Дубликаты не удаляются (`drop_duplicates: false`)

### Запуск через DVC

В проекте настроен pipeline из двух этапов: `prepare` и `train`

Если DVC установлен:

```powershell
dvc repro
```

Эта команда выполняет подготовку данных и обучение модели с учётом зависимостей и параметров

## Запуск API

Сначала нужно убедиться, что модель существует в `models/model.joblib`

Затем запустить API

```powershell
python -m uvicorn api.app:app --reload --port 8000
```

После запуска доступны:

- `GET http://127.0.0.1:8000/health` - проверка готовности модели
- `POST http://127.0.0.1:8000/predict` - получение предсказания
- `http://127.0.0.1:8000/docs` - интерактивная документация Swagger

Пример запроса для `/predict`:

```json
{
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
}
```

Пример ответа:

```json
{
  "predicted_quality": 5.81,
  "unit": "expert score (approx. 3–9)",
  "interpretation": "Среднее качество"
}
```

API проверяет входящие значения и использует сохранённую модель `models/model.joblib`

## Запуск пользовательского интерфейса

Нужно открыть второй терминал VS Code и активировать виртуальное окружение (терминал в котором Docker не закрывать, благодаря ему работает API)

```powershell
.\.venv\Scripts\Activate.ps1
```

Запустите Streamlit

```powershell
python -m streamlit run ui/app.py
```

Интерфейс откроется в браузере

Пользователь вводит физико-химические характеристики вина и нажимает кнопку предсказания

UI отправляет HTTP-запрос к API и отображает полученный результат, модель в самом интерфейсе не загружается

По умолчанию используется адрес API

```text
http://localhost:8000
```

При необходимости адрес можно изменить через переменную окружения `API_URL`

## Docker Compose

Перед сборкой Docker-образов необходимо подготовить данные и обучить модель:

```bash
dvc repro
```

Затем собрать и запустить оба сервиса

```bash
docker compose up --build
```

После запуска:

- UI (Streamlit): http://localhost:8501
- API (FastAPI): http://localhost:8000
- Документация API: http://localhost:8000/docs

Внутри Docker-сети UI обращается к API по адресу `http://api:8000`.

Для остановки контейнеров:

```bash
docker compose down
```
## Тестирование

Для запуска автоматических тестов:

```powershell
python -m pytest -q
```

Результат проверки проекта:

```text
3 passed, 1 warning
```

Все три теста API успешно пройдены

## Качество модели

Метрики получены на тестовой выборке из 980 объектов

| Метрика | Значение |
|---|---|
| MAE | 0.4832 |
| RMSE | 0.6385 |
| R² | 0.4807 |

Модель в среднем ошибается примерно на 0.48 балла экспертной оценки

Дополнительная информация находится в `metrics.json` и `reports/metrics_report.txt`

## Выводы EDA

- Пропуски в исходном датасете отсутствуют
- Около 19% строк являются дубликатами
- В проведённых экспериментах удаление дубликатов ухудшало MAE, поэтому они оставлены
- Распределение целевой переменной `quality` несбалансировано, чаще встречаются оценки 5 и 6
- Все признаки числовые, дополнительное кодирование категорий не требуется
- Масштабирование признаков для RandomForest не обязательно
- Наиболее значимым признаком по результатам анализа важности модели оказался `alcohol`

## Полный путь предсказания

```text
Пользователь
    ↓
Streamlit UI
    ↓ HTTP POST /predict
FastAPI
    ↓
models/model.joblib
    ↓
Предсказание quality
    ↓
JSON-ответ API
    ↓
Отображение результата в Streamlit
```

Проект включает подготовку данных, обучение и сохранение модели, API для получения предсказаний, отдельный пользовательский интерфейс и автоматические тесты
