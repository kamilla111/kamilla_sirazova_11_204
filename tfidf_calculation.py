import os
import re
import math
from collections import defaultdict, Counter
import pymorphy2

PAGES_DIR = "pages"
TFIDF_TERMS_DIR = "tfidf_terms"
TFIDF_LEMMAS_DIR = "tfidf_lemmas"

os.makedirs(TFIDF_TERMS_DIR, exist_ok=True)
os.makedirs(TFIDF_LEMMAS_DIR, exist_ok=True)

morph = pymorphy2.MorphAnalyzer()

STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничто', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между',
    'the', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 'these', 'those', 'it', 'its', 'their', 'our', 'we', 'you', 'he', 'she', 'they',
    'amp', 'nbsp', 'hellip', 'ndash', 'mdash', 'laquo', 'raquo', 'quot', 'apos', 'lt', 'gt', 'http', 'https', 'www', 'com', 'ru', 'org', 'net', 'io', 'co', 'tv'
}


def clean_html(html):
    html = re.sub(r'<script.*?</script>|<style.*?</style>|<!--.*?-->', '', html, flags=re.DOTALL | re.I)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


doc_term_counts = {}
doc_lemma_counts = {}
document_frequency_terms = defaultdict(int)
document_frequency_lemmas = defaultdict(int)

documents = []

for filename in sorted(os.listdir(PAGES_DIR)):
    if not filename.endswith(".html"):
        continue

    doc_id = filename.replace(".html", "")
    documents.append(doc_id)

    with open(os.path.join(PAGES_DIR, filename), encoding='utf-8') as f:
        html = f.read()

    text = clean_html(html)
    tokens = re.findall(r'\b[а-яё]+\b', text.lower())
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 3]

    term_counts = Counter(tokens)
    lemma_counts = Counter()

    for token, count in term_counts.items():
        lemma = morph.parse(token)[0].normal_form
        lemma_counts[lemma] += count

    doc_term_counts[doc_id] = term_counts
    doc_lemma_counts[doc_id] = lemma_counts

    for term in term_counts:
        document_frequency_terms[term] += 1

    for lemma in lemma_counts:
        document_frequency_lemmas[lemma] += 1

N = len(documents)


for doc_id in documents:

    term_counts = doc_term_counts[doc_id]
    lemma_counts = doc_lemma_counts[doc_id]

    total_terms = sum(term_counts.values())

    with open(f"{TFIDF_TERMS_DIR}/{doc_id}_terms.txt", "w", encoding="utf-8") as f:

        for term, freq in term_counts.items():
            tf = freq / total_terms
            idf = math.log(N / document_frequency_terms[term])
            tfidf = tf * idf
            f.write(f"{term} {idf:.6f} {tfidf:.6f}\n")

    with open(f"{TFIDF_LEMMAS_DIR}/{doc_id}_lemmas.txt", "w", encoding="utf-8") as f:

        for lemma, freq in lemma_counts.items():
            tf = freq / total_terms
            idf = math.log(N / document_frequency_lemmas[lemma])
            tfidf = tf * idf
            f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

print("TF-IDF рассчитан для всех документов.")