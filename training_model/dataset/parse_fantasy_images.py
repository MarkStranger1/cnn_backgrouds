# requests - для HTTP-запросов к API Pixabay,
# os - для работы с файловой системой,
# time - для ожидания между повторными запросами,
# tqdm - для визуализации прогресса загрузки.
import requests, os, time
from tqdm import tqdm

# Ключ API для доступа к Pixabay.
API_KEY = "52838265-7f75ba84cc80806d57cdd4244"

# Список поисковых запросов к Pixabay.
# Каждый элемент - кортеж: (поисковая строка, имя_файла_префикс)
# Например: ("fantasy forest", "forest") означает, что будут скачаны картинки по запросу “fantasy forest” и сохранены с префиксом “forest”.
queries = [
    ("fantasy forest", "forest"),
    ("fantasy mountain", "mountain"),
    ("fantasy river", "river"),
    ("fantasy castle", "castle"),
    ("fantasy city", "city"),
    ("sci fi landscape", "sci_fi_landscape"),
    ("sci fi city", "sci_fi_city"),
    ("sci fi sky", "sci_fi_sky"),
    ("cyberpunk landscape", "cyberpunk_landscape"),
]

# Папка, в которую будут сохранены загруженные изображения - training_model/dataset/raw_data/fantasy
output_dir = os.path.join("raw_data", "fantasy")
os.makedirs(output_dir, exist_ok=True)  # создаёт директорию, если её нет

# Основной цикл по каждому поисковому запросу
for query in queries:
    # Для каждого запроса загружается 5 страниц результатов (по 100 изображений на странице)
    for page in tqdm(range(1, 6), desc="Pages"):
        # Формируем URL-запрос к Pixabay API
        url = f"https://pixabay.com/api/?key={API_KEY}&q={query[0]}&image_type=illustration&per_page=100&page={page}"

        # Попытка загрузки данных с повторными запросами при ошибках (до 5 попыток)
        for attempt in range(5):
            try:
                data = requests.get(url, timeout=20).json()  # запрос и конвертация в JSON
                break  # если успешно, выходим из цикла попыток
            except Exception as e:
                print(f"Ошибка при загрузке страницы {page} (попытка {attempt+1}/5): {e}")
                time.sleep(2)  # ждём 2 секунды перед повтором
        else:
            # Если все 5 попыток неудачны - пропускаем страницу
            print(f"⚠️ Пропуск страницы {page}")
            continue

        # Перебор всех изображений, найденных на странице
        for i, hit in enumerate(data.get("hits", [])):
            # Формируем путь, по которому будет сохранено изображение
            img_path = os.path.join(output_dir, f"{query[1]}_{page}_{i}.jpg")

            # Если файл уже существует, пропускаем загрузку
            if os.path.exists(img_path):
                continue

            # Получаем ссылку на изображение большого размера
            img_url = hit["largeImageURL"]

            # Загружаем и сохраняем изображение
            try:
                img = requests.get(img_url, timeout=20).content  # скачиваем содержимое
                with open(img_path, "wb") as f:  # записываем байты в файл
                    f.write(img)
            except Exception as e:
                print(f"❌ Ошибка при загрузке изображения: {e}")
