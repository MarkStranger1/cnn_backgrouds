import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Параметры и пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Директория текущего скрипта
DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # Путь к папке с датасетом
MODEL_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))  # Корень проекта (на уровень выше)

IMG_SIZE = (224, 224)  # Размер изображений, используемых для модели
BATCH_SIZE = 32        # Размер пакета для генератора
SEED = 42              # Фиксируем случайность для воспроизводимости

np.random.seed(SEED)      # Фиксируем seed для numpy
tf.random.set_seed(SEED)  # Фиксируем seed для TensorFlow

# Генератор данных для валидации
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input  # Предобработка для ResNet50
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),  # Папка с валидационными изображениями
    target_size=IMG_SIZE,              # Изменение размера изображений
    batch_size=BATCH_SIZE,             # Размер пакета
    class_mode='categorical',          # Классификация по категориям
    shuffle=False,                     # Не перемешивать данные (важно для предсказаний)
    seed=SEED
)

# Загрузка предобученной модели
model_path = os.path.join(MODEL_ROOT, "avatar_bg_classifier_resnet50.h5")
print(f"Загрузка модели из {model_path} ...")
model = load_model(model_path)  # Загрузка модели из файла .h5

# Оценка модели на валидационном наборе
loss, acc = model.evaluate(val_gen, verbose=1)  # Вычисление loss и accuracy
print(f"\nValidation accuracy: {acc:.4f}")
print(f"Validation loss: {loss:.4f}\n")

#Предсказания на валидационных данных
print("Вычисление предсказаний...")
val_gen.reset()  # Сброс генератора, чтобы начать с первого батча
preds = model.predict(val_gen, verbose=1)  # Получение вероятностей для каждого класса
y_pred = np.argmax(preds, axis=1)  # Класс с максимальной вероятностью
y_true = val_gen.classes             # Истинные метки классов
class_labels = list(val_gen.class_indices.keys())  # Названия классов

# Отчёты о классификации
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))  # Основные метрики (precision, recall, f1-score)

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)  # Создание матрицы ошибок
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)  # Визуализация матрицы ошибок
plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# Визуализация примеров предсказаний
print("\nПримеры предсказаний:")
for i in range(5):  # Берём первый батч
    img, label = val_gen[i]  # Получаем изображения и истинные метки
    preds = model.predict(img)  # Предсказания модели
    for j in range(3):  # Показываем первые 3 изображения из батча
        true_label = class_labels[np.argmax(label[j])]  # Истинный класс
        pred_label = class_labels[np.argmax(preds[j])]  # Предсказанный класс
        confidence = preds[j][np.argmax(preds[j])] * 100  # Вероятность предсказанного класса
        plt.imshow((img[j] + 1) / 2)  # Визуализация изображения (масштабируем обратно к [0,1])
        plt.title(f"True: {true_label} | Pred: {pred_label} ({confidence:.1f}%)")
        plt.axis('off')
        plt.show()
    break  # Только первый батч
