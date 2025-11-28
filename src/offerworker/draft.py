import requests
import time

COORDINATOR_URL = "http://coordinator-service:8000"


def worker_routine():
    while True:
        try:
            resp = requests.get(f"{COORDINATOR_URL}/get_task").json()
        except Exception as e:
            print(e)
            time.sleep(5)
            continue

        if resp["status"] == "empty":
            break

        start_id = resp["start_id"]
        end_id = resp["end_id"]

        print(f"Обрабатываю диапазон {start_id} - {end_id}")

        # --- ВАША ЛОГИКА 3 и 4 ЭТАПА ---
        # 2. SELECT * FROM users WHERE user_id >= start_id AND user_id < end_id
        # 3. Генерация эмбеддингов для этих юзеров
        # 4. Поиск в FAISS (индекс Активных Юзеров должен быть уже загружен в память воркера!)
        # 5. INSERT результатов в таблицу рекомендаций
        # -------------------------------


if __name__ == "__main__":
    worker_routine()
