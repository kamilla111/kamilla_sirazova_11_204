# Deployment Manual

# Клонирование репозитория
```bash
git clone https://github.com/kamilla111/kamilla_sirazova_11_204.git
cd kamilla_sirazova_11_204
```

# Установка зависимостей
```bash
pip install requests pymorphy2
```

# Подготовка входных данных
В корне проекта должен находиться файл `urls.txt` (128 ссылок на русском языке).

# Запуск программы

**Шаг 1 — Скачивание страниц**
```bash
python crawler.py
```

**Шаг 2 — Токенизация и лемматизация (задание 2)**
```bash
python process.py
```
**Шаг 3 — Построение индекса и поиск (задание 3)**
```bash
python search.py
```
После запуска search.py можно вводить поисковые запросы в консоль. 

**Шаг 4 — Расчёт TF-IDF**
```bash
python tfidf.py
```

# Результат выполнения
После выполнения скриптов в проекте будут созданы:

- `pages/` — скачанные HTML-файлы (1.html, 2.html, …)
- `tokens/` — токены **по каждой странице** (`1_tokens.txt`, `2_tokens.txt`, …)
- `lemmas/` — леммы **по каждой странице** (`1_lemmas.txt`, `2_lemmas.txt`, …)
- `index.txt` — соответствие номера страницы и URL
- `tokens.txt` — **общий список токенов**
- `lemmas.txt` — **группировка токенов по леммам**
- `inverted_index.json` — инвертированный индекс
- `tfidf_terms/` — TF-IDF по терминам 
- `tfidf_lemmas/` — TF-IDF по леммам
- `tfidf.py` — расчёт TF-IDF (Задание 4)






