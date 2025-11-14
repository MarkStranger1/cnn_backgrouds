import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------- 
# Параметры 
# ---------

# Путь к текущему файлу (папка с val_model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Путь к обученной модели (на уровень выше, в папке model/)
MODEL_PATH = os.path.join(BASE_DIR, "..\\model\\avatar_bg_classifier_resnet50.h5")

# JSON-файл с метаданными пользователей и путями к изображениям
JSON_PATH = os.path.join(BASE_DIR, "VK_dataset\\filtered_users.json")

# Базовая директория, где лежат все изображения из VK_dataset
IMG_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "VK_dataset"))

# Размер входного изображения для модели
IMG_SIZE = (224, 224)

# Список классов в порядке, соответствующем выходу модели
CLASSES = ['fantasy', 'landscape', 'urban']

# Загружаем предобученную модель ResNet50 (fine-tuned на нашем датасете)
model = load_model(MODEL_PATH)


# -------------
# Загрузка JSON 
# -------------

# Открываем JSON с информацией о пользователях и их фото
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Преобразуем JSON в DataFrame для удобной итерации и анализа
df = pd.DataFrame(data)


# ----------------------------------
# Функция предобработки с блюром лиц 
# ----------------------------------

def preprocess_img(relative_path, target_size=IMG_SIZE):
    """
    Загружает изображение по относительному пути, размывает лица (анонимизация),
    изменяет размер под модель и выполняет предобработку ResNet50.
    """
    # Полный путь к файлу изображения
    full_path = os.path.join(IMG_BASE_DIR, relative_path)
    print(f"Пытаемся открыть: {full_path}")

    # Загружаем изображение с диска
    img = cv2.imread(full_path)
    if img is None:
        # Если файл не найден — выбрасываем ошибку
        raise ValueError(f"Файл не найден: {full_path}")

    # Переводим изображение из BGR (OpenCV) в RGB (для корректного отображения)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Детекция лиц ---
    # Используем встроенный каскад Хаара из OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # находим лица

    # Для каждого найденного лица применяем сильное размытие (анонимизация)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        img[y:y+h, x:x+w] = face  # заменяем область лица на размытый вариант

    # --- resize и препроцессинг ---
    # Изменяем размер до (224, 224), как требует ResNet50
    img = cv2.resize(img, target_size)

    # Добавляем измерение batch (1, 224, 224, 3)
    x = np.expand_dims(img, axis=0)

    # Применяем стандартную предобработку ResNet50 (нормализация пикселей в диапазон, ожидаемый моделью)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

# ----------------------------------
# Предсказания модели с уверенностью
# ----------------------------------

# Сюда сохраняются результаты: (id пользователя, класс, уверенность)
predictions = []

# tqdm — индикатор прогресса обработки всех записей
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predictions"):
    try:
        # 1. Загружаем и обрабатываем изображение
        x = preprocess_img(row['path_photo_240'])

        # 2. Получаем предсказание от модели (вектор вероятностей)
        pred = model.predict(x, verbose=0)

        # 3. Определяем класс с максимальной вероятностью
        pred_class = CLASSES[np.argmax(pred)]

        # 4. Сохраняем максимальное значение вероятности (уверенность модели)
        confidence = float(np.max(pred))

        # Добавляем результат в список
        predictions.append((row['id'], pred_class, confidence))

    except Exception as e:
        # Если возникла ошибка при обработке файла (например, нет изображения)
        print(f"Ошибка при обработке {row['id']}: {e}")
        # Добавляем "пустой" результат для сохранения структуры
        predictions.append((row['id'], None, None))


# Преобразуем список предсказаний в DataFrame для удобства анализа
pred_df = pd.DataFrame(predictions, columns=['id', 'pred_class', 'confidence'])

# Объединяем результаты предсказаний с исходными данными пользователей по ключу 'id'
df = df.merge(pred_df, on='id')


# -------------------------------
# Общая оценка уверенности модели 
# -------------------------------

# Берём только валидные значения уверенности (без NaN)
valid_confidences = df['confidence'].dropna()

# Если есть хотя бы одно корректное предсказание — считаем статистику
if len(valid_confidences) > 0:
    avg_conf = valid_confidences.mean()       # средняя уверенность модели
    median_conf = valid_confidences.median()  # медианная уверенность
    min_conf = valid_confidences.min()        # минимальное значение
    max_conf = valid_confidences.max()        # максимальное значение

    # Выводим агрегированные показатели уверенности
    print(f"\n--- Оценка уверенности модели ---")
    print(f"Средняя уверенность: {avg_conf:.3f}")
    print(f"Медианная уверенность: {median_conf:.3f}")
    print(f"Мин. уверенность: {min_conf:.3f}")
    print(f"Макс. уверенность: {max_conf:.3f}")
else:
    # Если все предсказания пустые (ошибки или недоступные изображения)
    print("Нет корректных предсказаний для расчета уверенности.")


# ----------------------------------
# 10 случайных фото с предсказаниями 
# ----------------------------------

# Визуализируем 10 случайных выборок по 10 изображений каждая
for iteration in range(10):
    sample = df.sample(10)  # случайные 10 записей из таблицы
    plt.figure(figsize=(20,5))
    for i, (_, row) in enumerate(sample.iterrows()):
        # Полный путь к файлу изображения
        full_path = os.path.join(IMG_BASE_DIR, row['path_photo_240'])
        # Загружаем и конвертируем изображение из BGR → RGB
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Добавляем подграфик в сетку 2x5
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        # Заголовок каждого изображения — предсказанный класс и уверенность
        pred_class = row['pred_class']
        confidence = row['confidence']
        plt.title(f"{pred_class}\n{confidence:.2f}")
        plt.axis('off')  # убираем оси координат
    plt.tight_layout()
    plt.show()  # отображаем окно с изображениями


# ----------------------------
# Функция построения диаграммы
# ----------------------------

def plot_distribution(df_subset, filename, title):
    """
    Строит столбчатую диаграмму распределения предсказанных классов
    и сохраняет её в файл.
    """
    plt.figure(figsize=(6,4))
    # sns.countplot — строит гистограмму количества элементов каждого класса
    sns.countplot(x='pred_class', data=df_subset, order=CLASSES)
    plt.title(title)
    # Сохраняем диаграмму в PNG и закрываем, чтобы не накапливались открытые окна
    plt.savefig(filename)
    plt.close()


# -----------------
# Ответы на вопросы
# -----------------

# 1. Распределение типов аватаров у пользователей 18–35 лет
plot_distribution(df[df['age'].between(18,35)], "question1_youth.png", "Распределение аватаров 18-35 лет")

# 2. Гендерные различия (1 — female, 2 — male)
for sex, label in zip([1,2], ['female','male']):
    plot_distribution(df[df['sex']==sex], f"question2_{label}.png", f"Распределение аватаров, {label}")

# 3. Различия по типу образования
for edu in ['техническое', 'гуманитарное', 'естественнонаучное']:
    plot_distribution(df[df['education']==edu], f"question3_{edu}.png", f"Распределение аватаров, {edu} образование")

# 4. Сравнение общей выборки и подгруппы 18–35 лет
plot_distribution(df, "question4_all.png", "Общая выборка")
plot_distribution(df[df['age'].between(18,35)], "question4_youth.png", "Возраст 18-35")

# 5. Мужчины 18–35 лет
plot_distribution(df[(df['age'].between(18,35)) & (df['sex']==2)], "question5_male_youth.png", "Мужчины 18-35")

# 6. Женщины 18–35 лет
plot_distribution(df[(df['age'].between(18,35)) & (df['sex']==1)], "question6_female_youth.png", "Женщины 18-35")

# 7. Влияние образования на тип аватара
for edu in ['техническое', 'гуманитарное', 'естественнонаучное']:
    plot_distribution(df[df['education']==edu], f"question7_{edu}.png", f"Аватары и образование: {edu}")
