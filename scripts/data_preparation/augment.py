import pandas as pd
import json
from collections import Counter
import re
from openai import OpenAI, AsyncClient
import os
import time  # Для задержки между запросами
import asyncio
from tqdm.asyncio import tqdm as tqdm_async


folder_id = os.environ["folder_id"]
api_key = os.environ["api_key"]

model = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"
client = OpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id,
)

# Загрузка данных
try:
    df_llm_data = pd.read_csv(os.getenv("USER_FEATURES_FP"))
    print(f"Загружено {len(df_llm_data)} строк из CSV.")
except Exception as e:
    print(f"Ошибка: {e}")
    exit()

try:
    products_df = pd.read_csv("products.csv")
    product_list_formatted = []
    for index, row in products_df.iterrows():
        product_info = (
            f"- product_name: {row['product_name']}\n"
            f"  product_type: {row['product_type']}\n"
            f"  description: {row['description']}"
        )
        product_list_formatted.append(product_info)
    all_products_text = "\n\n".join(product_list_formatted)
except Exception as e:
    print(f"Ошибка: {e}")
    exit()


# Конвертация строковых представлений списков/кортежей в актуальные Python объекты
def parse_clickhouse_array(s):
    if pd.isna(s) or not s.strip():
        return []

    s = s.strip()
    if s == "{}":
        return []

    if "{" in s and "}" in s and s.count("{") > 1:
        elements_str = s.strip("{}")
        matches = re.findall(r"{([^}]+)}", elements_str)

        parsed_elements = []
        for match in matches:
            parts = [p.strip() for p in match.split(",")]
            if len(parts) == 2:
                try:
                    brand_id = int(parts[0]) if parts[0].strip() else None
                    price = float(parts[1])
                    parsed_elements.append((brand_id, price))
                except (ValueError, IndexError):
                    continue
        return parsed_elements

    elif s.startswith("{") and s.endswith("}"):
        elements = [
            el.strip().strip("'\" ")
            for el in s.strip("{}").split(",")
            if el.strip()
        ]
        return [el for el in elements if el]

    return []


df_llm_data["brand_spendings"] = df_llm_data["brand_spendings"].apply(
    parse_clickhouse_array
)
df_llm_data["categories_list"] = df_llm_data["categories_list"].apply(
    parse_clickhouse_array
)
df_llm_data["subcategories_list"] = df_llm_data["subcategories_list"].apply(
    parse_clickhouse_array
)

# Шаг 2.1.1: Сбор уникальных категорий и подкатегорий
unique_terms_set = set()
for _, row in df_llm_data.iterrows():
    unique_terms_set.update(row["categories_list"])
    unique_terms_set.update(row["subcategories_list"])

unique_terms = sorted(list(unique_terms_set))
print(
    f"Найдено {len(unique_terms)} уникальных категорий и подкатегорий для перевода."
)

# Шаг 2.1.2: Вызов LLM для перевода (с кешированием)
translation_cache_file = "translation_cache.json"
english_to_russian_map = {}

# Загружаем кеш, если существует
if os.path.exists(translation_cache_file):
    with open(translation_cache_file, "r", encoding="utf-8") as f:
        english_to_russian_map = json.load(f)
    print(f"Загружен кеш переводов. {len(english_to_russian_map)} записей.")

# Переводим только то, чего нет в кеше
terms_to_translate = [
    term for term in unique_terms if term not in english_to_russian_map
]
print(f"Требуется перевести {len(terms_to_translate)} новых терминов.")

if terms_to_translate:
    print("Начинаем перевод через OpenAI API. Это может занять время...")
    # Разделяем на батчи по 20-50 терминов, чтобы не превысить контекстное окно и уменьшить число запросов
    batch_size = 50
    for i in range(0, len(terms_to_translate), batch_size):
        batch = terms_to_translate[i : i + batch_size]
        prompt_translate = f"""Переведи следующие английские термины категорий и подкатегорий товаров на русский язык. Ответь в формате JSON, где ключ - это английский термин, а значение - его русский перевод. Сохрани оригинальные ключи.

Пример:
{{
  "Electronics": "Электроника",
  "Home Appliances": "Бытовая техника"
}}

Термины:
{json.dumps(batch, ensure_ascii=False)}
"""
        try:
            response = client.responses.create(
                model=model,
                instructions="You are a helpful assistant that translates category terms. answer with a valid json.",
                input=prompt_translate,
                # response_format={"type": "json_object"},
                temperature=0.1,  # Для точного перевода
            )

            translated_batch = json.loads(response.output_text)
            english_to_russian_map.update(translated_batch)
            print(
                f"Переведено {len(batch)} терминов. Всего в кеше: {len(english_to_russian_map)}"
            )
            time.sleep(0.5)  # Задержка для соблюдения rate limits
        except Exception as e:
            print(f"Ошибка при переводе батча {i}-{i+batch_size}: {e}")
            print(f"Батч: {batch}")
            time.sleep(5)  # Большая задержка при ошибке, чтобы избежать бана
            continue

    # Сохраняем обновленный кеш
    with open(translation_cache_file, "w", encoding="utf-8") as f:
        json.dump(english_to_russian_map, f, ensure_ascii=False, indent=2)
    print("Кеш переводов обновлен и сохранен.")


# Шаг 2.1.3: Функция для получения перевода
def get_translated_term(term):
    return english_to_russian_map.get(
        term, term
    )  # Если перевода нет, возвращаем оригинал


def generate_llm_prompt_v2(row):
    user_id = row["user_id"]
    region = (
        f"Регион {int(row['region'])}"
        if pd.notna(row["region"])
        else "неизвестном регионе"
    )
    socdem_cluster = (
        f"соц. кластер {int(row['socdem_cluster'])}"
        if pd.notna(row["socdem_cluster"])
        else "неизвестном соц. кластере"
    )

    # Разделяем общий поток средств на входящие и исходящие
    # Предполагаем, что Price в Payments может быть как положительным (поступление), так и отрицательным (расход)
    total_income_month = 0.0
    total_outcome_month = 0.0

    # Агрегация из brand_spendings, чтобы получить детали по поступлениям/расходам
    brand_spendings_map = Counter()
    for brand_id, price in row["brand_spendings"]:
        if brand_id is not None:
            if price > 0:
                total_income_month += price
            else:
                total_outcome_month += price  # Отрицательное значение
            brand_spendings_map[brand_id] += price

    total_income_month = round(total_income_month, 2)
    total_outcome_month = round(
        abs(total_outcome_month), 2
    )  # Берем модуль для отображения как "расходы"

    transactions_count_month = int(
        row["transactions_count_month"]
    )  # Количество транзакций в целом

    top_brands = [
        f"Бренд {brand_id} ({round(total_price, 2)} у.е.)"
        for brand_id, total_price in brand_spendings_map.most_common(5)
    ]
    top_brands_text = ", ".join(top_brands) if top_brands else "нет данных"

    # Переведенные категории и подкатегории
    translated_categories = [
        get_translated_term(c) for c in row["categories_list"] if c
    ]
    translated_subcategories = [
        get_translated_term(s) for s in row["subcategories_list"] if s
    ]

    top_categories = [
        f"{cat} ({count})"
        for cat, count in Counter(translated_categories).most_common(5)
    ]
    top_categories_text = (
        ", ".join(top_categories) if top_categories else "нет данных"
    )

    top_subcategories = [
        f"{subcat} ({count})"
        for subcat, count in Counter(translated_subcategories).most_common(5)
    ]
    top_subcategories_text = (
        ", ".join(top_subcategories) if top_subcategories else "нет данных"
    )

    prompt = f"""
Ты — продвинутый банковский аналитик с глубоким пониманием потребностей клиентов. Тебе предстоит проанализировать профиль клиента и предложить наиболее релевантные банковские продукты.

Профиль клиента:
- ID пользователя: {user_id}
- Социально-демографическая информация: Клиент находится в {region} и относится к {socdem_cluster}.
- Финансовая активность за последний месяц:
  - Совершено транзакций: {transactions_count_month}
  - Общий входящий поток средств: {total_income_month} у.е.
  - Общий исходящий поток средств (расходы): {total_outcome_month} у.е.
- Потребительские предпочтения (по данным чеков):
  - Топ-5 наиболее часто покупаемых категорий товаров: {top_categories_text}.
  - Топ-5 наиболее часто покупаемых подкатегорий товаров: {top_subcategories_text}.
- Расходы по брендам:
  - Топ-5 брендов по сумме движения средств: {top_brands_text}.


Ниже приведен список доступных банковских продуктов с их описаниями.
---
СПИСОК ПРОДУКТОВ:
{all_products_text}
---

На основе представленного профиля клиента и детального описания продуктов, определи, какие 2 продукта из списка выше наиболее релевантны для этого клиента в ближайшем будущем.

Твоя задача — дать максимально точную рекомендацию, основанную на паттернах поведения и потенциальных потребностях. В ответе верни `product_name` рекомендованных продуктов.
Ответь строго в формате JSON, содержащем массив строк с названиями (`product_name`) рекомендованных продуктов. Если ты не можешь дать рекомендацию или не видишь подходящих продуктов, верни пустой массив.

Пример ответа:
{{
  "recommended_products": ["Кредитная карта", "Вклад"]
}}
"""
    return prompt


# Генерируем промпты с переведенными категориями и разделением на доходы/расходы
df_llm_data["llm_prompt"] = df_llm_data.apply(generate_llm_prompt_v2, axis=1)

# Инициализация асинхронного клиента
# Используем те же параметры, что и для синхронного, но через AsyncOpenAI
async_client = AsyncClient(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=os.environ["api_key"],
    project=os.environ["folder_id"],
)

# Кеширование ответов LLM
llm_responses_cache_file = "llm_recommendations_cache.json"
llm_recommendations_cache = {}

# Загружаем кеш, если существует
if os.path.exists(llm_responses_cache_file):
    try:
        with open(llm_responses_cache_file, "r", encoding="utf-8") as f:
            llm_recommendations_cache = json.load(f)
        print(
            f"Загружен кеш ответов LLM. {len(llm_recommendations_cache)} записей."
        )
    except json.JSONDecodeError:
        print("Ошибка при чтении кеша LLM, инициализируем пустой кеш.")
        llm_recommendations_cache = {}


async def get_llm_recommendation(user_id, prompt):
    # Проверяем кеш перед отправкой запроса
    if str(user_id) in llm_recommendations_cache:
        # print(f"Используем кеш для пользователя {user_id}") # Отключено для уменьшения вывода
        return user_id, llm_recommendations_cache[str(user_id)]

    try:
        # Указания для модели могут быть в 'instructions'
        response = await async_client.responses.create(
            model=model,
            instructions="You are a helpful assistant that recommends banking products. Respond with a JSON object. If no products are suitable or data is insufficient, return an empty array.",
            input=prompt,
            temperature=0.1,  # Для более детерминированных ответов
        )

        # Парсинг ответа
        raw_output = response.output_text

        # Попытка извлечь JSON из текста, если модель не возвращает строго JSON
        json_match = re.search(
            r"```json\n(\{.*?\})\n```", raw_output, re.DOTALL
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = (
                raw_output  # Если нет ```json```, пробуем парсить весь текст
            )

        parsed_response = json.loads(json_str)
        recommended_products = parsed_response.get("recommended_products", [])

        # Кешируем ответ
        llm_recommendations_cache[str(user_id)] = recommended_products
        return user_id, recommended_products

    except json.JSONDecodeError as jde:
        print(
            f"Ошибка парсинга JSON для пользователя {user_id}: {jde}. Raw output: {raw_output[:500]}..."
        )
        # Сохраняем пустой список или индикатор ошибки
        llm_recommendations_cache[str(user_id)] = []
        return user_id, []
    except Exception as e:
        print(f"Ошибка LLM API для пользователя {user_id}: {e}")
        # При возникновении ошибки, сохраняем пустой список
        llm_recommendations_cache[str(user_id)] = []
        return user_id, []


async def process_all_prompts(df):
    tasks = []
    # Для каждого пользователя, если его промпт не пуст и он еще не в кеше
    # (проверка в get_llm_recommendation)
    for index, row in df.iterrows():
        user_id = row["user_id"]
        prompt = row["llm_prompt"]
        # Отправляем задачу только если user_id еще нет в кеше (или если мы хотим обновить)
        # На самом деле, проверка внутри get_llm_recommendation более эффективна
        tasks.append(get_llm_recommendation(user_id, prompt))

    print(f"Начинаем обработку {len(tasks)} промптов через Async LLM API.")

    # Ограничение параллельных запросов (например, до 100 одновременных)
    # Это позволяет избежать перегрузки API и исчерпания ресурсов
    concurrency_limit = 10

    # Используем asyncio.Semaphore для ограничения параллелизма
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def limited_task(task_coro):
        async with semaphore:
            return await task_coro

    limited_tasks = [limited_task(task) for task in tasks]

    # Запускаем все задачи и ждем их выполнения
    # return await asyncio.gather(*tasks) # Без ограничения параллелизма
    return await tqdm_async.gather(
        *limited_tasks, desc="Генерация рекомендаций LLM"
    )


# Запуск асинхронного процесса
print("Начало генерации рекомендаций LLM.")
# Проверим, какие user_id уже есть в кеше
cached_user_ids = set(llm_recommendations_cache.keys())
# Фильтруем DataFrame, чтобы обрабатывать только новых пользователей
df_to_process = df_llm_data[
    ~df_llm_data["user_id"].astype(str).isin(cached_user_ids)
]

if not df_to_process.empty:
    print(f"Будет обработано {len(df_to_process)} новых пользователей.")
    llm_results = asyncio.new_event_loop().run_until_complete(
        process_all_prompts(df_to_process)
    )

    # Обновляем кэш после каждой успешной обработки партии
    with open(llm_responses_cache_file, "w", encoding="utf-8") as f:
        json.dump(llm_recommendations_cache, f, ensure_ascii=False, indent=2)
    print("Кеш ответов LLM обновлен.")
else:
    print("Все пользователи уже есть в кеше LLM, пропускаем вызов API.")
    llm_results = []  # Если все в кеше, то нет новых результатов от API


# Интеграция результатов обратно в DataFrame
# Создаем DataFrame из кешированных результатов для легкой интеграции
final_recommendations_df = pd.DataFrame(
    [
        {"user_id": int(uid), "llm_recommendations": recs}
        for uid, recs in llm_recommendations_cache.items()
    ]
)

df_llm_data = df_llm_data.merge(
    final_recommendations_df, on="user_id", how="left"
)
