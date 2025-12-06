import os
import requests
import re


os.makedirs("blimp_master/data", exist_ok=True)


base_url = "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/"


api_url = "https://api.github.com/repos/alexwarstadt/blimp/contents/data"
response = requests.get(api_url)
files = response.json()


jsonl_files = [f['name'] for f in files if f['name'].endswith('.jsonl')]

print(f"найдено файлов: {len(jsonl_files)}")


for filename in jsonl_files:

    url = base_url + filename
    response = requests.get(url)
    
    if response.status_code == 200:
        filepath = os.path.join("blimp_master/data", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"сохранен: {filename}")
    else:
        print(f"ошибка: {filename}")

print(f"\nскачано {len(jsonl_files)} файлов")

