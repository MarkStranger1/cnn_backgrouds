#!/usr/bin/env python3

"""
Скрипт подготавливает изображения для трёх классов: landscape, urban, fantasy.
Выполняет:
- проверку и сбор файлов;
- анонимизацию лиц (размытие при необходимости);
- изменение размера изображений до 224x224;
- аугментации (случайные преобразования);
- разделение на train и val;
- сохранение результата в структуре dataset/train/<class> и dataset/val/<class>.
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import logging
import albumentations as A

# -----------------------------------
# Константы и вспомогательные функции
# -----------------------------------

# Допустимые расширения изображений - набор форматов, которые будут обрабатываться, поддерживаемые Pillow и OpenCV.
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def list_images(paths):
    """
    Сканирует переданные пути и возвращает список всех файлов изображений.
    Работает как с одиночными файлами, так и с целыми директориями.
    """
    imgs = []  # сюда будут собираться пути ко всем найденным изображениям

    # Цикл по каждому элементу из списка путей (могут быть как файлы, так и каталоги)
    for p in paths:
        p = Path(p)  # приводим путь к объекту pathlib.Path для удобной работы
        if not p.exists():  # если путь не существует - пропускаем
            continue

        # Если путь указывает на конкретный файл
        if p.is_file():
            # Проверяем расширение - нужно, чтобы это был один из поддерживаемых форматов
            if p.suffix.lower() in IMG_EXTS:
                imgs.append(str(p))  # добавляем путь в список (в строковом виде)
        else:
            # Если это директория - ищем внутри все файлы с подходящими расширениями.
            # Используем метод rglob('*{ext}'), который рекурсивно обходит все подпапки.
            for ext in IMG_EXTS:
                imgs += [str(x) for x in p.rglob(f'*{ext}')]
    # Возвращаем итоговый список всех найденных изображений
    return imgs


# ---------------------------
# Инициализация детектора лиц
# ---------------------------
def init_face_detector():
    """
    Загружает каскад Хаара для обнаружения лиц (OpenCV).
    Используется для размытия лиц на изображениях.
    """
    # Определяем путь к файлу каскада Хаара.
    # Path(__file__).parent - директория, где лежит текущий скрипт (dataset)
    # .parent.parent.parent - поднимаемся на три уровня вверх до корня проекта
    # other_models - папка, где хранится файл каскада.
    local_path = Path(__file__).parent.parent.parent / "other_models" / "haarcascade_frontalface_default.xml"

    # Если файл не найден - выбрасываем исключение с пояснением
    if not local_path.exists():
        raise FileNotFoundError(f"Не найден файл каскада: {local_path}")

    # Загружаем XML-файл в каскадный классификатор OpenCV. Классификатор умеет искать лица на изображении, используя заранее обученные параметры.
    detector = cv2.CascadeClassifier(str(local_path))
    return detector  # возвращаем готовый объект детектора


def blur_faces_opencv(img_bgr, detector, factor=0.6):
    """
    Размывает найденные лица на изображении с помощью фильтра Гаусса.

    img_bgr: входное изображение в формате BGR (numpy)
    detector: объект cv2.CascadeClassifier для поиска лиц
    factor: коэффициент, определяющий силу размытия (0–1)
    """
    # Переводим изображение в оттенки серого для детектора лиц, т.к. он обучен на черно-белых данных
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Поиск лиц на изображении с помощью метода detectMultiScale()
    # scaleFactor=1.1 - уменьшение изображения на 10% на каждом уровне пирамиды (для разных масштабов лиц)
    # minNeighbors=4 - минимальное число соседних прямоугольников, подтверждающих лицо (чем выше, тем строже)
    # minSize=(30,30) - минимальный размер обнаруживаемого лица
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    # Если лиц нет - возвращаем исходное изображение без изменений и 0 найденных лиц
    if len(faces) == 0:
        return img_bgr, 0

    # Перебираем все найденные лица
    for (x, y, w, h) in faces:
        # Координаты области лица: (x, y) - верхний левый угол, (w, h) - ширина и высота
        # Проверяем, чтобы координаты не выходили за границы изображения
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_bgr.shape[1], x + w)
        y2 = min(img_bgr.shape[0], y + h)

        # Вырезаем участок изображения (Region of Interest - ROI), содержащий лицо
        roi = img_bgr[y1:y2, x1:x2]

        # Размер ядра размытия (kernel) подбирается в зависимости от размера лица
        # min(w, h) * factor - чем больше лицо, тем сильнее размытие
        # | 1 - операция побитового ИЛИ, которая гарантирует, что число будет нечётным
        # (для фильтра Гаусса требуется нечётное ядро)
        k = max(3, int(min(w, h) * factor) | 1)

        # Применяем размытие Гаусса. Оно сглаживает пиксели в области ROI, скрывая лицо
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        # Подменяем участок лица в исходном изображении на размытый фрагмент
        img_bgr[y1:y2, x1:x2] = blurred

    # Возвращаем обработанное изображение и количество найденных лиц
    return img_bgr, len(faces)


# ----------------------
# Работа с изображениями
# ----------------------

def load_image(path):
    """
    Безопасно загружает изображение с диска.
    Возвращает numpy-массив RGB или None при ошибке.
    """
    try:
        # Открываем изображение с помощью библиотеки Pillow.
        img = Image.open(path).convert('RGB') # гарантирует, что у всех изображений будет одинаковый формат.

        # Преобразуем Pillow-объект в массив NumPy для дальнейшей обработки
        return np.array(img)

    except Exception as e:
        # Если произошла любая ошибка, функция возвращает None, чтобы вызывающая сторона могла пропустить это изображение.
        return None


def get_augmentations():
    """
    Возвращает набор аугментаций (случайных преобразований),
    выполняемых с помощью библиотеки Albumentations.
    Эти операции помогают искусственно увеличить разнообразие данных,
    улучшая обобщающую способность нейросети.
    """
    tr = A.Compose([
        # Случайное обрезание и ресайз
        # Изображение масштабируется и вырезается случайный участок 224x224 пикселей.
        # scale=(0.8, 1.0) - выбирается участок, занимающий от 80% до 100% исходного размера.
        # p=0.7 - вероятность применения трансформации (70% изображений подвергнутся ей).
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.7),

        # Случайное горизонтальное отражение
        # p=0.5 - половина изображений будет перевёрнута по горизонтали.
        A.HorizontalFlip(p=0.5),

        # Смещение, масштабирование и вращение изображения.
        # shift_limit=0.05 - сдвиг до 5% размера кадра,
        # scale_limit=0.1 - увеличение/уменьшение до 10%,
        # rotate_limit=15 - поворот до 15 градусов.
        # p=0.5 - применяется к половине изображений.
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),

        # Случайная яркость и контраст.
        # p=0.5 - для разнообразия освещения в обучающем наборе.
        A.RandomBrightnessContrast(p=0.5),

        # Имитирует сжатие JPEG, чтобы повысить устойчивость модели к артефактам.
        # Понижает качество до диапазона 75–95.
        # p=0.3 - применяется с вероятностью 30%.
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),
    ])
    return tr


def save_image(img_arr, out_path, quality=92):
    """
    Сохраняет изображение в формате JPEG по указанному пути.
    Создаёт директории при необходимости.
    """
    # Преобразуем массив NumPy обратно в изображение Pillow (тип Image.Image).
    # Входной массив должен быть в RGB-формате
    pil = Image.fromarray(img_arr)

    # Убеждаемся, что родительская папка существует.
    # Параметр parents=True создаёт всю иерархию, если её нет.
    # exist_ok=True - не выдаёт ошибку, если каталог уже существует.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем изображение в формате JPEG:
    # - quality=92 - баланс между размером и качеством (по умолчанию у Pillow 75);
    # - optimize=True - оптимизирует таблицы JPEG для уменьшения размера файла.
    pil.save(str(out_path), format='JPEG', quality=quality, optimize=True)


# ---------------------------------------------------
# Основная логика подготовки данных для одного класса
# ---------------------------------------------------
def prepare_class(class_name, sources, out_dir, n_target, val_ratio, face_detector, augmentor, face_blur=True, seed=42):
    """
    Подготавливает изображения для одного класса.
    Шаги:
    1. Собирает все изображения из sources;
    2. Перемешивает их;
    3. Делит на train/val;
    4. Опционально размывает лица;
    5. Делает resize и crop до 224x224;
    6. Генерирует аугментированные версии (только для train);
    7. Сохраняет всё в соответствующие папки.
    """
    # Фиксируем seed для воспроизводимости - чтобы при повторных запусках порядок shuffle был одинаковым
    random.seed(seed)

    # Список всех путей к изображениям, найденных в директориях sources
    all_imgs = list_images(sources)
    logging.info(f"[{class_name}] Found {len(all_imgs)} images in sources: {sources}")

    # Если файлов нет - пишем предупреждение и выходим
    if len(all_imgs) == 0:
        logging.warning(f"[{class_name}] No images found for class {class_name}.")
        return 0, 0

    # Перемешиваем изображения, чтобы train/val получились случайными
    random.shuffle(all_imgs)

    # Подсчёт, сколько изображений пойдёт в валидацию и обучение
    n_val = int(n_target * val_ratio)   # например, 20% валидация
    n_train = n_target - n_val          # остальные 80% - train

    # Папки, куда сохраняются обработанные изображения
    train_out = Path(out_dir) / 'train' / class_name
    val_out = Path(out_dir) / 'val' / class_name

    # Счётчики - сколько изображений уже подготовлено
    produced_train = 0
    produced_val = 0

    # Индекс текущего источника (из all_imgs)
    i_src = 0

    # Номер итерации, если все исходные изображения уже исчерпаны и нужно использовать их повторно для аугментаций
    augment_iter = 0

    # tqdm - индикатор прогресса (отображает в консоли ход выполнения)
    pbar = tqdm(total=n_target, desc=f'Preparing {class_name}', unit='img')

    # -----------------------------------
    # Основной цикл генерации изображений
    # -----------------------------------
    while produced_train + produced_val < n_target:
        # Если все исходные изображения уже использованы - снова перемешиваем список и начинаем с начала.
        # Это позволяет «дополнить» датасет до нужного числа за счёт аугментаций.
        if i_src >= len(all_imgs):
            i_src = 0
            random.shuffle(all_imgs)
            logging.info(f"[{class_name}] Re-iterating source images for augmentation (iter {augment_iter})")
            augment_iter += 1

        # Берём очередной путь к изображению
        src_path = all_imgs[i_src]
        i_src += 1

        # Загружаем изображение в numpy-массив
        img = load_image(src_path)
        # Если файл битый или не поддерживается - пропускаем
        if img is None:
            continue

        # Переводим изображение из RGB в BGR, потому что OpenCV работает в BGR.
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Если включено face_blur=True - применяем размытие лиц
        if face_blur:
            img_bgr, nfaces = blur_faces_opencv(img_bgr, face_detector)

        # Возвращаем изображение обратно в RGB (для корректного сохранения через Pillow)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Определяем, куда идёт текущее изображение - train или val
        target_slot = 'train' if produced_train < n_train else 'val'
        out_base = train_out if target_slot == 'train' else val_out

        # Базовая обработка изображения - resize + центрирование до 224×224
        try:
            # Получаем оригинальные размеры
            h0, w0 = img.shape[:2]

            # Сначала масштабируем короткую сторону до 256 пикселей, чтобы сохранить пропорции, но уменьшить разброс размеров.
            short = 256
            scale = short / min(h0, w0)
            new_w = int(round(w0 * scale))
            new_h = int(round(h0 * scale))

            # Изменяем размер с помощью билинейной интерполяции
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Центрируем изображение и вырезаем квадрат 224×224
            start_x = (new_w - 224) // 2
            start_y = (new_h - 224) // 2
            img_cropped = img_resized[start_y:start_y+224, start_x:start_x+224]

            # Сохранение базового изображения
            idx = produced_train + produced_val + 1  # порядковый номер файла
            out_path = out_base / f'{class_name}_{idx:06d}.jpg'  # формат имени: fantasy_000123.jpg
            save_image(img_cropped, out_path)  # сохраняем

            # Обновляем счётчики в зависимости от того, куда шло изображение
            if target_slot == 'train':
                produced_train += 1
            else:
                produced_val += 1

            # Обновляем прогресс-бар
            pbar.update(1)

        except Exception as e:
            # Если с каким-то изображением возникла ошибка - пропускаем
            logging.debug(f"Error processing base image {src_path}: {e}")
            continue

        # Генерация аугментированных изображений (только для train)
        if target_slot == 'train' and produced_train < n_train:
            # Сколько изображений ещё нужно для train
            remaining = n_train - produced_train

            # Сколько аугментаций создадим из текущего изображения
            # Обычно не более 2, чтобы не перегружать процесс (можно регулировать)
            n_aug = min(2, remaining)

            for k in range(n_aug):
                try:
                    # Применяем аугментации к исходному изображению
                    aug = augmentor(image=img)
                    aug_img = aug['image']

                    # Проверяем, что размер совпадает с 224×224 (иногда augmentor может вернуть чуть другой размер)
                    if aug_img.shape[0] != 224 or aug_img.shape[1] != 224:
                        aug_img = cv2.resize(aug_img, (224, 224), interpolation=cv2.INTER_AREA)

                    # Генерируем имя для нового файла
                    idx = produced_train + produced_val + 1
                    out_path = out_base / f'{class_name}_{idx:06d}.jpg'

                    # Сохраняем аугментированное изображение
                    save_image(aug_img, out_path)
                    produced_train += 1  # увеличиваем счётчик
                    pbar.update(1)

                    # Если достигли нужного количества изображений - прекращаем цикл
                    if produced_train + produced_val >= n_target:
                        break

                except Exception as e:
                    # Ошибки аугментации
                    logging.debug(f"Augmentation error for {src_path}: {e}")
                    continue

    # Завершаем обработку
    pbar.close()
    logging.info(f"[{class_name}] produced train={produced_train}, val={produced_val}")
    return produced_train, produced_val


# --------------------------
# Загрузка датасета с Kaggle
# --------------------------
def download_kaggle(dataset, out_dir):
    """
    Скачивает датасет с Kaggle по его slug-имени (например, 'username/dataset-name').
    Требует предварительно настроенный API-ключ (файл kaggle.json в ~/.kaggle/).
    """
    import subprocess  # Модуль для запуска системных команд из Python

    # Формируем команду для CLI-интерфейса Kaggle:
    # kaggle datasets download -d <dataset> -p <папка> --unzip
    # -d <dataset> : идентификатор набора данных на Kaggle (user/dataset)
    # -p <папка>   : куда сохранить файлы
    # --unzip      : автоматически распаковать после скачивания
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', out_dir, '--unzip']

    # subprocess.check_call(cmd) выполняет команду как будто в терминале:
    # если код возврата != 0 (ошибка) - вызовет исключение и остановит выполнение.
    subprocess.check_call(cmd)


# -----------------------------------
# Парсинг аргументов командной строки
# -----------------------------------
def parse_args():
    # Создаём парсер аргументов, который позволяет запускать скрипт
    # с разными параметрами через терминал (пример - ниже).
    #
    # Пример вызова:
    # python prepare_dataset.py \
    #   --sources_landscape ./raw_data/landscape \
    #   --sources_urban ./raw_data/urban \
    #   --sources_fantasy ./raw_data/fantasy \
    #   --out_dir ./dataset \
    #   --n_per_class 4000 \
    #   --val_ratio 0.2 \
    #   --face_blur
    p = argparse.ArgumentParser()

    # Пути к папкам с исходными изображениями для каждого класса.
    # nargs='*' означает, что можно передать несколько путей подряд.
    # Пример: --sources_landscape ./set1 ./set2 ./set3
    p.add_argument('--sources_landscape', nargs='*', default=[], help='Пути к исходным изображениям класса landscape')
    p.add_argument('--sources_urban', nargs='*', default=[], help='Пути к исходным изображениям класса urban')
    p.add_argument('--sources_fantasy', nargs='*', default=[], help='Пути к исходным изображениям класса fantasy')

    # Папка, в которую будут сохраняться train/val. По умолчанию - ./dataset (в той же директории, где лежит скрипт)
    p.add_argument('--out_dir', type=str, default='./dataset', help='Папка для сохранения подготовленного датасета')

    # Сколько изображений нужно подготовить для каждого класса (всего, включая train и val)
    p.add_argument('--n_per_class', type=int, default=4000, help='Количество изображений на класс (train + val)')

    # Пропорция для разделения train/val (0.2 → 20% данных пойдут в validation)
    p.add_argument('--val_ratio', type=float, default=0.2, help='Доля данных для валидации')

    # Флаг, включающий размытие лиц. Если указан (просто флаг без значения) - True.
    p.add_argument('--face_blur', action='store_true', help='Размывать лица, если указано')

    # Фиксированный seed для детерминированного перемешивания
    p.add_argument('--seed', type=int, default=42)

    # Необязательный аргумент - можно указать список Kaggle-датасетов для скачивания.
    # Пример: --download_kaggle user1/setA user2/setB
    p.add_argument('--download_kaggle', nargs='*', default=[], help='Необязательные датасеты Kaggle для скачивания')

    # Возвращаем объект со всеми параметрами в виде атрибутов (args.sources_landscape и т.д.)
    return p.parse_args()


# ----------------
# Основная функция
# ----------------
def main():
    """
    Главная функция - управляет всем процессом подготовки датасета.
    Шаги:
    1. Настраивает логирование;
    2. Читает аргументы командной строки;
    3. Определяет пути к исходным данным (raw_data);
    4. Инициализирует инструменты (детектор лиц, аугментации);
    5. Запускает обработку для каждого класса;
    6. Считает статистику и выводит результат.
    """

    # ---------------------
    # Настройка логирования
    # ---------------------
    # Устанавливает уровень логов INFO (будут видны основные сообщения). Формат лога: "время | уровень | сообщение".
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # --------------------------------------
    # Получаем аргументы из командной строки
    # --------------------------------------
    # Вызов parse_args() возвращает объект args со всеми параметрами (например, args.n_per_class, args.val_ratio, args.face_blur и т.д.)
    args = parse_args()

    # ---------------------------------
    # Определяем пути к исходным данным
    # ---------------------------------
    # Все исходные данные лежат в поддиректории raw_data/<class> относительно текущего файла prepare_dataset.py
    # Path(__file__).parent - путь к текущей папке (dataset)
    raw_data_dir = Path(__file__).parent / "raw_data"

    # Задаём пути к исходным изображениям каждого класса вручную:
    args.sources_landscape = [str(raw_data_dir / "landscape")]
    args.sources_urban = [str(raw_data_dir / "urban")]
    args.sources_fantasy = [str(raw_data_dir / "fantasy")]

    # -----------------------------------------------------------------
    # Определяем выходную директорию (куда сохранять готовые train/val)
    # -----------------------------------------------------------------
    # В проекте train и val находятся в той же папке, что и скрипт (dataset/)
    args.out_dir = str(Path(__file__).parent)

    # --------------------------
    # Инициализация инструментов
    # --------------------------
    # Если включено размытие лиц - создаём детектор. В противном случае просто None (для ускорения).
    face_detector = init_face_detector() if args.face_blur else None

    # Создаём набор аугментаций
    augmentor = get_augmentations()

    # Убеждаемся, что выходная папка существует
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Подготавливаем список классов и соответствующих путей
    # -----------------------------------------------------
    # Каждый элемент - кортеж (имя_класса, список_источников)
    classes_and_sources = [
        ('landscape', args.sources_landscape),
        ('urban', args.sources_urban),
        ('fantasy', args.sources_fantasy),
    ]

    # ---------------------------------
    # Обработка всех классов по очереди
    # ---------------------------------
    total = 0  # счётчик общего числа созданных изображений (train + val)

    for cls, srcs in classes_and_sources:
        # Проверяем, что для класса действительно указаны источники
        if not srcs:
            logging.warning(f"No source provided for class {cls}. Skipping.")
            continue

        # Запускаем основную функцию обработки для текущего класса (создание train/val, resize, аугментации, face blur и т.д.)
        t_train, t_val = prepare_class(
            class_name=cls,
            sources=srcs,
            out_dir=out_dir,
            n_target=args.n_per_class,
            val_ratio=args.val_ratio,
            face_detector=face_detector,
            augmentor=augmentor,
            face_blur=args.face_blur,
            seed=args.seed
        )

        # Прибавляем количество сгенерированных изображений
        total += t_train + t_val

    # --------------------
    # Завершаем выполнение
    # --------------------
    # Печатаем итоговые данные в лог
    logging.info(f"Done. Total images produced: {total}")

    # Дублируем сообщение в консоль
    print("Finished preparing dataset.")


# Точка входа
if __name__ == '__main__':
    main()
