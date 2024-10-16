import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Ustawienie zmiennej środowiskowej, aby API szukało pliku w folderze projektu
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')  # Zmieniona ścieżka, bez ukośnika na początku
os.environ['KAGGLE_USERNAME'] = 'aleksanderignacik'  # np. 'john_doe'
os.environ['KAGGLE_KEY'] = '1f9218f8383175887b7f88bd58132752'  # np. 'abcdef1234567890abcdef1234567890'

# Inicjalizacja i autoryzacja API
api = KaggleApi()
api.authenticate()

# Ustawienie folderu, gdzie będą przechowywane dane
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Pobierz dane z konkursu (Upewnij się, że nazwa konkursu jest poprawna w API)
api.competition_download_files('child-mind-institute-problematic-internet-use', path=data_dir)

# Rozpakowanie danych (jeśli są spakowane w ZIP)
import zipfile
zip_file_path = os.path.join(data_dir, 'child-mind-institute-problematic-internet-use.zip')
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

print("Dane pobrane i rozpakowane!")
