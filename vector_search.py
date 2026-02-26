import os
import re
import math
from collections import Counter
import pymorphy2

TFIDF_LEMMAS_DIR = "tfidf_lemmas"
INDEX_TXT = "index.txt"

morph = pymorphy2.MorphAnalyzer()

STOP_WORDS = {
    'и', 'в', 'во', 'не',
    'что', 'он', 'на', 'я',
    'с', 'со', 'как', 'а',
    'то', 'все', 'она', 'так',
    'его', 'но', 'да', 'ты',
    'к', 'у', 'же', 'вы',
    'за', 'бы', 'по',
    'только', 'ее', 'мне',
    'было', 'вот', 'от',
    'меня', 'еще', 'нет', 'о',
    'из', 'ему', 'теперь',
    'когда', 'даже', 'ну',
    'вдруг', 'ли', 'если',
    'уже', 'или', 'ни',
    'быть', 'был', 'него',
    'до', 'вас', 'нибудь',
    'опять', 'уж', 'вам',
    'ведь', 'там', 'потом',
    'себя', 'ничто', 'ей',
    'может', 'они', 'тут',
    'где', 'есть', 'надо',
    'ней', 'для', 'мы',
    'тебя', 'их', 'чем',
    'была', 'сам', 'чтоб',
    'без', 'будто', 'чего',
    'раз', 'тоже', 'себе',
    'под', 'будет', 'ж',
    'тогда', 'кто', 'этот',
    'того', 'потому', 'этого',
    'какой', 'совсем', 'ним',
    'здесь', 'этом', 'один',
    'почти', 'мой', 'тем',
    'чтобы', 'нее', 'сейчас',
    'были', 'куда', 'зачем',
    'всех', 'никогда',
    'можно', 'при', 'наконец',
    'два', 'об', 'другой',
    'хоть', 'после', 'над',
    'больше', 'тот', 'через',
    'эти', 'нас', 'про',
    'всего', 'них', 'какая',
    'много', 'разве', 'три',
    'эту', 'моя', 'впрочем',
    'хорошо', 'свою', 'этой',
    'перед', 'иногда',
    'лучше', 'чуть', 'том',
    'нельзя', 'такой', 'им',
    'более', 'всегда',
    'конечно', 'всю', 'между',
    'the', 'and', 'or', 'but',
    'if', 'in', 'on', 'at',
    'to', 'of', 'for', 'with',
    'by', 'from', 'as', 'is',
    'are', 'was', 'were',
    'be', 'have', 'has',
    'had', 'do', 'does',
    'did', 'this', 'that',
    'these', 'those', 'it',
    'its', 'their', 'our',
    'we', 'you', 'he', 'she',
    'they',
    'amp', 'nbsp', 'hellip',
    'ndash', 'mdash', 'laquo',
    'raquo', 'quot', 'apos',
    'lt', 'gt', 'http',
    'https', 'www', 'com',
    'ru', 'org', 'net', 'io',
    'co', 'tv'
}


def clean_and_lemmatize(text):
    tokens = re.findall(
        r'\b[а-яёa-z]+\b',
        text.lower())
    lemmas = []
    for token in tokens:
        if token in STOP_WORDS or len(
                token) < 2:
            continue
        lemma = \
        morph.parse(token)[
            0].normal_form
        if lemma not in STOP_WORDS:
            lemmas.append(
                lemma)
    return lemmas


def load_tfidf_data():
    """Загружает все векторы документов и словарь IDF"""
    doc_vectors = {}
    idf_dict = {}

    for filename in sorted(
            os.listdir(
                    TFIDF_LEMMAS_DIR)):
        if not filename.endswith(
                "_lemmas.txt"):
            continue
        doc_id = filename.replace(
            "_lemmas.txt", "")
        vector = {}

        with open(
                os.path.join(
                        TFIDF_LEMMAS_DIR,
                        filename),
                encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    lemma, idf_str, tfidf_str = parts
                    tfidf = float(
                        tfidf_str)
                    idf = float(
                        idf_str)
                    vector[
                        lemma] = tfidf
                    if lemma not in idf_dict:  # IDF одинаковый для всех документов
                        idf_dict[
                            lemma] = idf
        doc_vectors[
            doc_id] = vector

    print(
        f"Загружено {len(doc_vectors)} документов, словарь: {len(idf_dict)} лемм")
    return doc_vectors, idf_dict


def query_to_vector(query,
                    idf_dict):
    """Превращает запрос в TF-IDF вектор"""
    lemmas = clean_and_lemmatize(
        query)
    if not lemmas:
        return {}

    term_counts = Counter(
        lemmas)
    total = len(lemmas)
    q_vector = {}

    for lemma, count in term_counts.items():
        if lemma in idf_dict:
            tf = count / total
            idf = idf_dict[
                lemma]
            q_vector[
                lemma] = tf * idf
    return q_vector


def cosine_similarity(vec1,
                      vec2):
    """Косинусное сходство"""
    if not vec1 or not vec2:
        return 0.0

    dot = 0.0
    for term, val1 in vec1.items():
        val2 = vec2.get(term,
                        0.0)
        dot += val1 * val2

    norm1 = math.sqrt(sum(
        v * v for v in
        vec1.values()))
    norm2 = math.sqrt(sum(
        v * v for v in
        vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (
                norm1 * norm2)


def load_url_map():
    """Загружает соответствие doc_id → URL"""
    url_map = {}
    if os.path.exists(
            INDEX_TXT):
        with open(INDEX_TXT,
                  encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(
                        maxsplit=1)
                    if len(parts) == 2:
                        fname, url = parts
                        doc_id = fname.replace(
                            ".html",
                            "")
                        url_map[
                            doc_id] = url
    return url_map


if __name__ == "__main__":
    print(
        "Загрузка векторного индекса...")
    doc_vectors, idf_dict = load_tfidf_data()
    url_map = load_url_map()

    print(
        "\nВекторный поиск (TF-IDF + Cosine Similarity)")
    print(
        "Введите запрос или 'exit' для выхода\n")

    while True:
        query = input(
            "Запрос: ").strip()
        if query.lower() in (
        "exit", "выход",
        "quit"):
            break
        if not query:
            continue

        q_vector = query_to_vector(
            query, idf_dict)

        results = []
        for doc_id, d_vector in doc_vectors.items():
            score = cosine_similarity(
                q_vector,
                d_vector)
            if score > 0:
                url = url_map.get(
                    doc_id,
                    f"pages/{doc_id}.html")
                results.append(
                    (score,
                     doc_id,
                     url))

        results.sort(
            reverse=True)  # по убыванию score

        print(
            f"\nНайдено релевантных документов: {len(results)}")
        print("Топ-15:")
        for i, (score, doc_id,
                url) in enumerate(
                results[:15],
                1):
            print(
                f"{i:2d}. [{doc_id}] score={score:.4f} → {url}")
        print("-" * 80)