import time

import numpy as np
import onnxruntime as ort


def main():
    print("Инициализация высокопроизводительного движка ONNX Runtime...")
    # 1. Загружаем файл модели в рантайм
    session = ort.InferenceSession("iris_model.onnx")

    # Узнаём, какой вход ожидает модель
    input_name = session.get_inputs()[0].name
    print(f"Модель ожидает входной тензор с именем: '{input_name}'")

    # 2. Тестовые данные: два цветка по 4 признака.
    # Тип строго float32 — граф ONNX жёстко типизирован
    dummy_input = np.array(
        [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]], dtype=np.float32
    )
    print(f"Входной вектор данных:\n{dummy_input}")

    # 3. Запускаем инференс и замеряем время
    start = time.perf_counter()
    raw_predictions = session.run(None, {input_name: dummy_input})
    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted_classes = raw_predictions[0]
    probabilities = raw_predictions[1]

    print("\nРезультаты инференса от ONNX Runtime:")
    print(f"Предсказанные классы для объектов: {predicted_classes}")
    print(f"Матрица вероятностей классов:\n{probabilities}")
    print(f"Время инференса: {elapsed_ms:.3f} мс")


if __name__ == "__main__":
    main()