from dataclasses import dataclass, field
from typing import Optional
import json
import re
import os
import time
from collections import Counter
from tqdm import tqdm
from openai import OpenAI


@dataclass
class PromptInfo:
    user_id: int
    socdem_cluster: float
    region: float
    total_spent: float
    categories: list
    subcategories: list


@dataclass
class Product:
    product_id: str
    product_type: str
    product_name: str
    description: str


@dataclass
class UserRecommendation:
    user_id: int
    recommended_products: list[Product] = field(default_factory=list)
    error: Optional[str] = None


class RecommendationGenerator:
    def __init__(
        self,
        products_file: str = "products.csv",
        translation_cache_file: str = "translation_cache.json",
        recommendations_cache_file: str = "llm_recommendations_cache.json",
    ):
        self.folder_id = os.environ["folder_id"]
        self.api_key = os.environ["api_key"]
        self.model = f"gpt://{self.folder_id}/qwen3-235b-a22b-fp8/latest"
        
        self.client = OpenAI(
            base_url="https://rest-assistant.api.cloud.yandex.net/v1",
            api_key=self.api_key,
            project=self.folder_id,
        )
        
        self.translation_cache_file = translation_cache_file
        self.recommendations_cache_file = recommendations_cache_file
        
        # Загружаем продукты как список словарей и создаём lookup
        self.products_list: list[Product] = []
        self.products_by_name: dict[str, Product] = {}
        self.all_products_text = self._load_products(products_file)
        
        # Загружаем кэши
        self.translation_cache = self._load_json_cache(translation_cache_file)
        self.recommendations_cache = self._load_json_cache(recommendations_cache_file)
    
    def _load_products(self, products_file: str) -> str:
        """Загрузка списка продуктов из CSV."""
        try:
            import csv
            product_list_formatted = []
            
            with open(products_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Создаём объект Product
                    product = Product(
                        product_id=row.get('product_id', ''),
                        product_type=row.get('product_type', ''),
                        product_name=row.get('product_name', ''),
                        description=row.get('description', '')
                    )
                    self.products_list.append(product)
                    
                    # Создаём lookup по имени (нормализованному)
                    normalized_name = product.product_name.strip().lower()
                    self.products_by_name[normalized_name] = product
                    
                    # Форматируем для промпта
                    product_info = (
                        f"- product_id: {product.product_id}\n"
                        f"  product_name: {product.product_name}\n"
                        f"  product_type: {product.product_type}\n"
                        f"  description: {product.description}"
                    )
                    product_list_formatted.append(product_info)
            
            print(f"Загружено {len(self.products_list)} продуктов.")
            return "\n\n".join(product_list_formatted)
            
        except Exception as e:
            print(f"Ошибка загрузки продуктов: {e}")
            return ""
    
    def _find_product_by_name(self, product_name: str) -> Optional[Product]:
        """Поиск продукта по имени (с нечётким соответствием)."""
        normalized_name = product_name.strip().lower()
        
        # Точное совпадение
        if normalized_name in self.products_by_name:
            return self.products_by_name[normalized_name]
        
        # Частичное совпадение
        for key, product in self.products_by_name.items():
            if normalized_name in key or key in normalized_name:
                return product
        
        return None
    
    def _enrich_recommendations(self, product_names: list[str]) -> list[Product]:
        """Обогащение списка названий продуктов полной информацией."""
        enriched = []
        not_found = []
        
        for name in product_names:
            product = self._find_product_by_name(name)
            if product:
                enriched.append(product)
            else:
                not_found.append(name)
                # Создаём Product с минимальной информацией
                enriched.append(Product(
                    product_id="UNKNOWN",
                    product_type="UNKNOWN",
                    product_name=name,
                    description=""
                ))
        
        if not_found:
            print(f"  Не найдены продукты: {not_found[:5]}{'...' if len(not_found) > 5 else ''}")
        
        return enriched
    
    def _load_json_cache(self, filepath: str) -> dict:
        """Загрузка JSON-кэша из файла."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                print(f"Загружен кэш из {filepath}. {len(cache)} записей.")
                return cache
            except json.JSONDecodeError:
                print(f"Ошибка чтения кэша {filepath}, инициализируем пустой.")
        return {}
    
    def _save_json_cache(self, cache: dict, filepath: str):
        """Сохранение JSON-кэша в файл."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    
    def _translate_terms(self, terms: list[str]) -> None:
        """Перевод терминов через LLM API."""
        terms_to_translate = [
            term for term in terms 
            if term and term not in self.translation_cache
        ]
        
        if not terms_to_translate:
            return
        
        print(f"Требуется перевести {len(terms_to_translate)} новых терминов.")
        
        batch_size = 50
        for i in range(0, len(terms_to_translate), batch_size):
            batch = terms_to_translate[i:i + batch_size]
            prompt_translate = f"""Переведи следующие английские термины категорий и подкатегорий товаров на русский язык. Ответь в формате JSON, где ключ - это английский термин, а значение - его русский перевод.

Пример:
{{
  "Electronics": "Электроника",
  "Home Appliances": "Бытовая техника"
}}

Термины:
{json.dumps(batch, ensure_ascii=False)}
"""
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions="You are a helpful assistant that translates category terms. Answer with a valid JSON.",
                    input=prompt_translate,
                    temperature=0.1,
                )
                
                translated_batch = json.loads(response.output_text)
                self.translation_cache.update(translated_batch)
                print(f"Переведено {len(batch)} терминов.")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Ошибка при переводе батча {i}-{i+batch_size}: {e}")
                time.sleep(5)
                continue
        
        self._save_json_cache(self.translation_cache, self.translation_cache_file)
        print("Кэш переводов обновлён.")
    
    def _get_translated_term(self, term: str) -> str:
        """Получение перевода термина."""
        return self.translation_cache.get(term, term)
    
    def _generate_prompt(self, info: PromptInfo) -> str:
        """Генерация промпта для пользователя."""
        region = (
            f"Регион {int(info.region)}"
            if info.region and not (isinstance(info.region, float) and info.region != info.region)
            else "неизвестном регионе"
        )
        
        socdem_cluster = (
            f"соц. кластер {int(info.socdem_cluster)}"
            if info.socdem_cluster and not (isinstance(info.socdem_cluster, float) and info.socdem_cluster != info.socdem_cluster)
            else "неизвестном соц. кластере"
        )
        
        # Переведённые категории и подкатегории
        translated_categories = [
            self._get_translated_term(c) for c in info.categories if c
        ]
        translated_subcategories = [
            self._get_translated_term(s) for s in info.subcategories if s
        ]
        
        top_categories = [
            f"{cat} ({count})"
            for cat, count in Counter(translated_categories).most_common(5)
        ]
        top_categories_text = ", ".join(top_categories) if top_categories else "нет данных"
        
        top_subcategories = [
            f"{subcat} ({count})"
            for subcat, count in Counter(translated_subcategories).most_common(5)
        ]
        top_subcategories_text = ", ".join(top_subcategories) if top_subcategories else "нет данных"
        
        total_spent_formatted = round(info.total_spent, 2) if info.total_spent else 0.0
        
        prompt = f"""
Ты — продвинутый банковский аналитик с глубоким пониманием потребностей клиентов. Тебе предстоит проанализировать профиль клиента и предложить наиболее релевантные банковские продукты.

Профиль клиента:
- ID пользователя: {info.user_id}
- Регион: {region}
- Социально-демографический кластер: {socdem_cluster}
- Финансовая активность за последний месяц:
  - Общая сумма расходов: {total_spent_formatted} у.е.
- Потребительские предпочтения (по данным чеков):
  - Топ-5 наиболее часто покупаемых категорий товаров: {top_categories_text}.
  - Топ-5 наиболее часто покупаемых подкатегорий товаров: {top_subcategories_text}.

Ниже приведен список доступных банковских продуктов с их описаниями.
---
СПИСОК ПРОДУКТОВ:
{self.all_products_text}
---

На основе представленного профиля клиента и детального описания продуктов, определи, какие топ 30-60 продукта из списка выше наиболее релевантны для этого клиента в ближайшем будущем.

Твоя задача — дать максимально точную рекомендацию, основанную на паттернах поведения и потенциальных потребностях. В ответе верни `product_name` рекомендованных продуктов.
Ответь строго в формате JSON, содержащем массив строк с названиями (`product_name`) рекомендованных продуктов. Если ты не можешь дать рекомендацию или не видишь подходящих продуктов, верни пустой массив.

Пример ответа:
{{
  "recommended_products": ["Кредитная карта", "Вклад"]
}}
"""
        return prompt
    
    def _get_recommendation_from_llm(self, user_id: int, prompt: str) -> UserRecommendation:
        """Получение рекомендации от LLM для одного пользователя."""
        cache_key = str(user_id)
        
        # Проверяем кэш
        if cache_key in self.recommendations_cache:
            cached_names = self.recommendations_cache[cache_key]
            enriched_products = self._enrich_recommendations(cached_names)
            return UserRecommendation(
                user_id=user_id,
                recommended_products=enriched_products
            )
        
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions="You are a helpful assistant that recommends banking products. Respond with a JSON object. If no products are suitable or data is insufficient, return an empty array.",
                input=prompt,
                temperature=0.1,
            )
            
            raw_output = response.output_text
            
            # Попытка извлечь JSON из текста
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*\"recommended_products\".*\}", raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = raw_output
            
            parsed_response = json.loads(json_str)
            product_names = parsed_response.get("recommended_products", [])
            
            # Кэшируем только названия (для совместимости)
            self.recommendations_cache[cache_key] = product_names
            
            # Обогащаем полной информацией
            enriched_products = self._enrich_recommendations(product_names)
            
            return UserRecommendation(
                user_id=user_id,
                recommended_products=enriched_products
            )
            
        except json.JSONDecodeError as jde:
            error_msg = f"JSON parse error: {jde}"
            print(f"Ошибка парсинга JSON для пользователя {user_id}: {jde}")
            self.recommendations_cache[cache_key] = []
            return UserRecommendation(user_id=user_id, error=error_msg)
            
        except Exception as e:
            error_msg = f"API error: {e}"
            print(f"Ошибка LLM API для пользователя {user_id}: {e}")
            self.recommendations_cache[cache_key] = []
            return UserRecommendation(user_id=user_id, error=error_msg)
    
    def generate_recommendations(
        self, 
        users_data: list[PromptInfo],
        delay_between_requests: float = 0.5,
        save_cache_every: int = 10,
    ) -> list[UserRecommendation]:
        """
        Генерация рекомендаций для списка пользователей.
        
        Args:
            users_data: Список PromptInfo с данными пользователей
            delay_between_requests: Задержка между запросами (секунды)
            save_cache_every: Сохранять кэш каждые N запросов
            
        Returns:
            Список UserRecommendation с рекомендациями (полная информация о продуктах)
        """
        # Шаг 1: Собираем уникальные термины для перевода
        all_terms = set()
        for info in users_data:
            all_terms.update(info.categories)
            all_terms.update(info.subcategories)
        
        # Шаг 2: Переводим термины
        self._translate_terms(list(all_terms))
        
        # Шаг 3: Фильтруем пользователей, которых нет в кэше
        users_to_process = [
            info for info in users_data 
            if str(info.user_id) not in self.recommendations_cache
        ]
        
        print(f"Всего пользователей: {len(users_data)}")
        print(f"Уже в кэше: {len(users_data) - len(users_to_process)}")
        print(f"Будет обработано: {len(users_to_process)}")
        
        results = []
        
        # Добавляем результаты из кэша (с обогащением)
        for info in users_data:
            cache_key = str(info.user_id)
            if cache_key in self.recommendations_cache:
                cached_names = self.recommendations_cache[cache_key]
                enriched_products = self._enrich_recommendations(cached_names)
                results.append(UserRecommendation(
                    user_id=info.user_id,
                    recommended_products=enriched_products
                ))
        
        # Обрабатываем новых пользователей
        if users_to_process:
            for i, info in enumerate(tqdm(users_to_process, desc="Генерация рекомендаций")):
                prompt = self._generate_prompt(info)
                recommendation = self._get_recommendation_from_llm(info.user_id, prompt)
                results.append(recommendation)
                
                # Сохраняем кэш периодически
                if (i + 1) % save_cache_every == 0:
                    self._save_json_cache(
                        self.recommendations_cache, 
                        self.recommendations_cache_file
                    )
                
                # Задержка между запросами
                if i < len(users_to_process) - 1:
                    time.sleep(delay_between_requests)
            
            # Финальное сохранение кэша
            self._save_json_cache(
                self.recommendations_cache, 
                self.recommendations_cache_file
            )
            print("Кэш рекомендаций обновлён.")
        
        return results
    
    def recommendations_to_dict(self, recommendations: list[UserRecommendation]) -> list[dict]:
        """Конвертация рекомендаций в список словарей для сериализации."""
        result = []
        for rec in recommendations:
            rec_dict = {
                "user_id": rec.user_id,
                "error": rec.error,
                "recommended_products": [
                    {
                        "product_id": p.product_id,
                        "product_type": p.product_type,
                        "product_name": p.product_name,
                        "description": p.description
                    }
                    for p in rec.recommended_products
                ]
            }
            result.append(rec_dict)
        return result
