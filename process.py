import os
import re
from collections import defaultdict
import pymorphy2

PAGES_DIR = "pages"
TOKENS_DIR = "tokens"
LEMMAS_DIR = "lemmas"

os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(LEMMAS_DIR, exist_ok=True)

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


all_tokens = set()
global_lemma_groups = defaultdict(set)


for filename in sorted(os.listdir(PAGES_DIR)):
    if not filename.endswith(".html"):
        continue

    page_id = filename.replace(".html", "")
    path = os.path.join(PAGES_DIR, filename)

    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()

    text = clean_html(html)

    tokens_raw = re.findall(r'\b[а-яё]+\b', text.lower())

    page_tokens = {t for t in tokens_raw
                   if t not in STOP_WORDS and len(t) >= 3}

    sorted_page_tokens = sorted(page_tokens)

    with open(f"{TOKENS_DIR}/{page_id}_tokens.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_page_tokens) + '\n')


    page_lemma_groups = defaultdict(list)
    for token in sorted_page_tokens:
        lemma = morph.parse(token)[0].normal_form
        page_lemma_groups[lemma].append(token)
        global_lemma_groups[lemma].add(token)
        all_tokens.add(token)


    with open(f"{LEMMAS_DIR}/{page_id}_lemmas.txt", 'w', encoding='utf-8') as f:
        for lemma in sorted(page_lemma_groups):
            toks = sorted(set(page_lemma_groups[lemma]))
            f.write(f"{lemma} {' '.join(toks)}\n")

    print(f"✓ Страница {page_id} — {len(page_tokens)} токенов")


# tokens.txt
with open('tokens.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sorted(all_tokens)) + '\n')

# lemmas.txt
with open('lemmas.txt', 'w', encoding='utf-8') as f:
    for lemma in sorted(global_lemma_groups):
        toks = sorted(global_lemma_groups[lemma])
        f.write(f"{lemma} {' '.join(toks)}\n")


print(f"Уникальных токенов всего: {len(all_tokens)}")
print(f"Групп лемм всего: {len(global_lemma_groups)}")
