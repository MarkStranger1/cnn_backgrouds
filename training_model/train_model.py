import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Параметры
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # путь к текущему файлу (training_model)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # путь к датасету (training_model/dataset)
MODEL_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))  # путь к корню проекта (на уровень выше)

IMG_SIZE = (224, 224)   # размер входных изображений для модели
BATCH_SIZE = 32         # размер батча
NUM_CLASSES = 3         # количество классов (urban, landscape, fantasy)
SEED = 42               # фиксированный seed для воспроизводимости

# Фиксируем случайность для всех используемых библиотек
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Генератор для обучающей выборки с аугментацией
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,  # предобработка под ResNet50
    rotation_range=20,              # случайные повороты до 20 градусов
    width_shift_range=0.1,          # горизонтальные сдвиги до 10% ширины
    height_shift_range=0.1,         # вертикальные сдвиги до 10% высоты
    shear_range=0.1,                # сдвиги по углу (shear transform)
    zoom_range=0.15,                # масштабирование до ±15%
    horizontal_flip=True,           # случайное отражение по горизонтали
    brightness_range=(0.8, 1.2),    # случайное изменение яркости
    fill_mode='nearest'             # заполнение пустых пикселей ближайшими значениями
)

# Генератор для валидационной выборки (без аугментаций)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Создание потока данных из папки train/
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),   # путь к train/
    target_size=IMG_SIZE,                 # resize всех изображений до 224x224
    batch_size=BATCH_SIZE,                # количество изображений в одном батче
    class_mode='categorical',             # one-hot кодирование классов
    shuffle=True,                         # перемешивание данных перед каждой эпохой
    seed=SEED                             # фиксированный seed для стабильности
)

# Создание потока данных из папки val/
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),     # путь к val/
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,                        # без перемешивания - важно для стабильной оценки
    seed=SEED
)

# Модель: ResNet50 backbone + head
# Загружаем базовую предобученную модель ResNet50 без верхней части (include_top=False)
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet', # используем веса, обученные на ImageNet
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling=None
)

# Замораживаем веса базовой части - они не будут обучаться на первом этапе
base_model.trainable = False

# Создаём входной слой
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# Пропускаем данные через ResNet50 (feature extractor)
x = base_model(inputs, training=False)
# Преобразуем 4D тензор в 2D (усредняем пространственные признаки)
x = layers.GlobalAveragePooling2D()(x)
# Нормализуем признаки для стабилизации обучения
x = layers.BatchNormalization()(x)
# Полносвязный слой с 256 нейронами
x = layers.Dense(256, activation='relu')(x)
# Dropout для уменьшения переобучения
x = layers.Dropout(0.4)(x)
# Выходной слой - 3 класса с softmax
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Собираем итоговую модель
model = models.Model(inputs, outputs)

# Компилируем модель - оптимизатор Adam и кросс-энтропия
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Сохраняем лучшую модель по метрике val_accuracy
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_ROOT, "best_head.h5"), monitor='val_accuracy', save_best_only=True, mode='max'
)
# Раннее завершение обучения, если точность не растёт 6 эпох подряд
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
# Понижаем learning rate, если val_loss не улучшается 3 эпохи
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Обучаем только верхнюю часть модели (ResNet заморожен)
history_head = model.fit(
    train_gen,
    epochs=15,                              # количество эпох обучения "головы"
    validation_data=val_gen,                # валидация на val/
    callbacks=[checkpoint, es, reduce_lr]
)

# Размораживаем часть базовой модели для дообучения
base_model.trainable = True
fine_tune_at = 140                      # начиная с этого слоя разрешаем обучение
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False             # нижние слои остаются замороженными
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True              # верхние - обучаются

# Компилируем модель заново для тонкой настройки
model.compile(
    optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Новый checkpoint и колбэки для этапа дообучения
checkpoint_ft = callbacks.ModelCheckpoint(
    os.path.join(MODEL_ROOT, "best_finetuned.h5"), monitor='val_accuracy', save_best_only=True, mode='max'
)
es_ft = callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
reduce_lr_ft = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Дообучение
history_ft = model.fit(
    train_gen,
    epochs=20,                           # количество эпох дообучения
    validation_data=val_gen,
    callbacks=[checkpoint_ft, es_ft, reduce_lr_ft]
)

# Сохраняем полностью обученную модель (ResNet50 + голова) в корень проекта
model.save(os.path.join(MODEL_ROOT, "avatar_bg_classifier_resnet50.h5"))
