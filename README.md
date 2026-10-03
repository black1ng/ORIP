# Лабораторная работа №1. Трекинг экспериментов и версионирование артефактов

Дисциплина: «Основы развертывания интеллектуальных приложений».

Датасет Iris (153 строки, 4 признака, 3 класса) версионируется в DVC,
эксперименты логируются в MLflow, лучшая модель экспортируется в ONNX.

## Установка

```
python -m venv .venv
.venv\Scripts\activate
pip install dvc "mlflow==2.8.1" "sqlalchemy<2.1" scikit-learn pandas numpy onnx onnxruntime skl2onnx matplotlib
```

## Запуск

1. Получить датасет из DVC-хранилища: `dvc pull`
   (или создать заново: `python make_dataset.py`).
2. Запустить сервер MLflow:
   `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 127.0.0.1 --port 5000`
3. Обучить модель: `python train.py`
4. Зарегистрировать модель в Model Registry под именем `iris_production_model`
   и перевести в стадию Staging (через MLflow UI).
5. Экспортировать в ONNX: `python export_onnx.py`
6. Проверить инференс: `python inference_onnx.py`