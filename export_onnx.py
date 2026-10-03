from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

import mlflow
import mlflow.sklearn

# 1. Подключение к серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def main():
    # Формат: "models:/<имя_модели>/<стадия_или_версия>"
    model_uri = "models:/iris_production_model/Staging"
    print(f"Загрузка модели из реестра: {model_uri}...")

    # Скачиваем Python-объект модели из Model Registry
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Описываем вход: матрица из 4 признаков типа float32,
    # None означает любой размер батча
    initial_type = [("float_input", FloatTensorType([None, 4]))]

    print("Трансляция графа вычислений в формат ONNX...")
    onnx_model = to_onnx(
        loaded_model,
        initial_types=initial_type,
        target_opset=15,
        options={"zipmap": False},
    )

    # Сохраняем бинарный файл на диск
    onnx_filename = "iris_model.onnx"
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Экспорт успешно завершен! Файл сохранен как: '{onnx_filename}'")


main()