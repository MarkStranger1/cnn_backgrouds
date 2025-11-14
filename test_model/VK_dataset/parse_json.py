import ijson
import json
import os
import requests
from datetime import datetime
from tqdm import tqdm

# --- Основные константы ---
INPUT_FILE = 'users.json'             # Входной файл с полным списком пользователей (большой JSON)
OUTPUT_FILE = 'filtered_users.json'   # Файл, куда сохраняются отфильтрованные пользователи
PHOTOS_DIR = 'photos_240'             # Папка, куда будут скачаны фотографии
CURRENT_YEAR = datetime.now().year    # Текущий год — для вычисления возраста

# Создаём директорию для фото, если её нет
os.makedirs(PHOTOS_DIR, exist_ok=True)

# --- Ключевые слова для классификации образования ---
# Техническое направление (по подстрокам в названии факультета/кафедры)
TECH = [
    'инжен', 'электр', 'меха', 'айти', 'техн', 'автомат', 'робот', 'програм', 'систем',
    'строит', 'ради', 'энер', 'машиностро', 'комп', 'прибор', 'архитект', 'авиат', 'авто',
    'транспорт', 'электро', 'информ', 'связи', 'телеком', 'гидр', 'металл', 'оптик',
    'нанотех', 'материаловед', 'конструк', 'судостро', 'геодез', 'горн', 'логист', 'тепло',
    'двигат', 'химтех', 'техпроц', 'аэрокосм'
]

# Естественнонаучное направление
NATURAL = [
    'физ', 'хим', 'биол', 'экол', 'гео', 'геофиз', 'геолог', 'геодез', 'матем', 'мед',
    'агр', 'фарм', 'почв', 'генет', 'биотех', 'океан', 'атмосфер', 'астро', 'агрохим',
    'зоо', 'ветер', 'географ', 'статист', 'аналит', 'биоинж', 'микробиол', 'нейро'
]

# Гуманитарное направление
HUMAN = [
    'фил', 'язык', 'лингв', 'псих', 'педаг', 'соци', 'прав', 'юр', 'филос', 'эконом',
    'менедж', 'жур', 'полит', 'искус', 'дизайн', 'культур', 'религ', 'теат', 'муз',
    'истор', 'туризм', 'маркет', 'рекла', 'бизнес', 'бухгалт', 'аудит', 'управ', 'комм',
    'перевод', 'библиот', 'литерат', 'коммуник', 'социол', 'археол', 'сцен', 'гуманит'
]


def calc_age(bdate: str):
    """Вычисляет возраст пользователя, если дата рождения содержит год."""
    if not bdate:
        return None
    parts = bdate.split('.')
    # Ожидаемый формат даты: дд.мм.гггг
    if len(parts) == 3:
        try:
            year = int(parts[2])
            return CURRENT_YEAR - year
        except ValueError:
            # Если год не удалось преобразовать в число
            return None
    return None


def classify_education(user):
    """
    Определяет тип образования по ключевым словам из названия факультета/кафедры.
    Проверяется только первый университет в списке.
    """
    universities = user.get('universities')
    # Если информации об образовании нет — возвращаем None
    if not universities or not isinstance(universities, list) or len(universities) == 0:
        return None

    uni = universities[0]
    faculty = (uni.get('faculty_name') or '').lower()
    chair = (uni.get('chair_name') or '').lower()

    # Объединяем текст для поиска по ключевым словам
    text = f"{faculty} {chair}"

    # Проверяем по трём категориям ключевых слов
    if any(word in text for word in TECH):
        return "техническое"
    if any(word in text for word in NATURAL):
        return "естественнонаучное"
    if any(word in text for word in HUMAN):
        return "гуманитарное"

    # Если не найдено ни одного совпадения
    return None


def download_photo(url: str, user_id: str):
    """
    Скачивает фото по ссылке `url`, сохраняет в папку PHOTOS_DIR.
    Возвращает путь к локальному файлу или None при ошибке.
    """
    if not url or not user_id:
        return None
    filename = f"{user_id}.jpg"
    path = os.path.join(PHOTOS_DIR, filename)
    try:
        # Скачиваем изображение с таймаутом
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Сохраняем в файл
            with open(path, 'wb') as f:
                f.write(response.content)
            return path
    except Exception as e:
        print(f"⚠️ Ошибка при скачивании фото для {user_id}: {e}")
    return None


def process_big_json():
    """
    Главная функция обработки большого JSON-файла пользователей.
    Потоково (через ijson) извлекает записи, фильтрует их и сохраняет в новый JSON.
    """
    first_item = True  # флаг для корректного форматирования JSON (без запятой перед первым элементом)
    total, saved = 0, 0  # счётчики общего и сохранённого количества пользователей

    # Потоковое чтение входного JSON и запись результата
    with open(INPUT_FILE, 'rb') as f_in, open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        users = ijson.items(f_in, 'item')  # потоковое чтение элементов массива JSON
        f_out.write('[\n')  # начинаем JSON-массив вручную

        for user in tqdm(users, desc="Обработка пользователей", unit="usr"):
            total += 1  # увеличиваем счётчик общего количества

            # Пропускаем пользователей без фото
            if user.get('has_photo') != 1:
                continue

            # Получаем URL фото — предпочтительно более качественный
            url = user.get('photo_200_orig') or user.get('photo_max_orig')
            if not url:
                continue

            # Скачиваем фото и получаем путь к файлу
            local_path = download_photo(url, str(user.get('id')))
            if not local_path:
                continue  # если не удалось скачать фото

            # Определяем тип образования
            education_type = classify_education(user)

            # Формируем краткую запись для выходного JSON
            entry = {
                'id': user.get('id'),
                'age': calc_age(user.get('bdate')),
                'sex': user.get('sex'),
                'path_photo_240': local_path,
                'education': education_type
            }

            # Записываем в файл, добавляя запятые между элементами
            if not first_item:
                f_out.write(',\n')
            json.dump(entry, f_out, ensure_ascii=False)
            first_item = False
            saved += 1

        # Закрываем JSON-массив
        f_out.write('\n]\n')

    # Итоговая статистика
    print(f"\n✅ Готово! Сохранено {saved} пользователей из {total}. Фото в '{PHOTOS_DIR}'")


# --- Точка входа ---
if __name__ == '__main__':
    # Запускаем процесс фильтрации и подготовки данных
    process_big_json()
