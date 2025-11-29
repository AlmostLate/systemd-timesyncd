import os
import time
import logging
import requests
from typing import List, Tuple, Protocol, Dict, Any
import clickhouse_connect
from lib.score import User, UsedOffer, calculate_score_for_user
from lib.normalized_faiss import NormalizedFAISS
from synth_generation import RecommendationGenerator, Product

from dataclasses import dataclass

@dataclass
class PromptInfo:
    user_id: int
    socdem_cluster: float
    region: float
    total_spent: float
    categories: list
    subcategories: list


# --- CONFIGURATION ---
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")
CH_HOST = os.getenv("CH_HOST", "localhost")
CH_PORT = os.getenv("CH_PORT", "8123")
POLL_INTERVAL = 2  # Seconds to wait before retrying connection
RETRY_DELAY = 5  # Seconds to wait on failure

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("OfferWorker")


class InferenceModel(Protocol):
    """
    Protocol defining how the worker interacts with the ML library.
    Implement this interface in your actual library later.
    """

    def load_resources(self) -> None:
        """Load FAISS indices, models, etc."""
        ...

    def process_batch(
        self, user_rows: List[Tuple]
    ) -> List[Tuple[int, int, float]]:
        """
        Input: List of rows from ClickHouse (users table).
        Output: List of tuples (user_id, offer_id, score).
        """
        ...

class HistoricalModel:
    """
    Protocol defining how the worker interacts with the ML library.
    Implement this interface in your actual library later.
    """
    def __init__(self):
        self.ch_client = None
    def load_resources(self) -> None:
        """Load FAISS indices, models, etc."""
        self.generator = RecommendationGenerator(
            products_file="products.csv",
            translation_cache_file="translation_cache.json",
            recommendations_cache_file="llm_recommendations_cache.json"
        )
        self.offers_space_dim = len(self.generator.products_by_name)

    def process_batch(
        self, user_rows: List[Tuple]
    ) -> List[Tuple[int, int, float]]:
        """
        Input: List of rows from ClickHouse (users table).
        Output: List of tuples (user_id, offer_id, score).
        """
        """Establish ClickHouse connection with retry logic."""
        while not self.ch_client:
            try:
                self.ch_client = clickhouse_connect.get_client(
                    host=CH_HOST, port=CH_PORT, username='', password=''
                )
                # Verify connection
                self.ch_client.query("SELECT 1")
                logger.info(f"Connected to ClickHouse at {CH_HOST}:{CH_PORT}")
            except Exception as e:
                logger.error(
                    f"Failed to connect to ClickHouse: {e}. Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
        
        user_row = user_rows[0]
        user = User(int(user_row[0]), float(user_row[1]),  float(user_row[2]), [])

        prompt_infos: list[PromptInfo] = []
        prompt_info = self.get_prompt_info_for_user(user.uid)
        prompt_infos.append(prompt_info)
        query = f"""  
        SELECT user_id, count() as cnt
        FROM payments
        GROUP BY user_id
        HAVING cnt > 15 -- Высокоактивные пользователи
        ORDER BY 2 desc
        LIMIT 500
        """
        active_users_rows = self.ch_client.query(query).result_rows
        for active_user_row in active_users_rows:
            lead_user_uid = active_user_row[0]
            prompt_info = self.get_prompt_info_for_user(lead_user_uid)
            prompt_infos.append(prompt_info)
        
        recommendations = self.generator.generate_recommendations(
            users_data=prompt_infos,
            delay_between_requests=0.5,
            save_cache_every=10
        )

        rec_dict: Dict[int, list[Product]] = dict()
        for rec in recommendations:
            if rec.error is not None:
                continue
            rec_dict[rec.user_id] = rec.recommended_products
        

        query = f"""
        SELECT max(timestamp) - 30 * 24 * 3600 * 1000000 as ts_cutoff, max(timestamp) as ts_last
        FROM receipts
        WHERE user_id = {user.uid}
        """
        ts_window_row = self.ch_client.query(query).result_rows[0]
        min_ts, max_ts = ts_window_row[0], ts_window_row[1]
        window_time = max_ts - min_ts
        users: list[User] = []
        for pi in prompt_infos:
            if pi.user_id not in rec_dict:
                continue
            recs = rec_dict[pi.user_id]

            used_offers: list[UsedOffer] = []
            time_segments = 0
            total_time_segments = len(recs)

            for rec in recs:
                used_offers.append(
                    UsedOffer(
                        rec.product_id,
                        1 + min_ts + int(window_time*(time_segments / total_time_segments))
                    )
                )
                time_segments += 1
            users.append(User(
                pi.user_id,
                pi.socdem_cluster,
                pi.region,
                used_offers,
            ))
    
        norm_faiss = NormalizedFAISS(self.offers_space_dim)
        records: Dict[int, List[int]] = []

        users_dict: Dict[int, User] = dict()
        for user in users:
            records[user.uid] = [uo.offer_id for uo in user.used_offers]
            users_dict[user.uid] = user

        norm_faiss.build(records)

        inclusions = norm_faiss.find_top_inclusions(0, 25, 200) # list of offerd_ids

        top_lead_users = []
        for inclusion in inclusions:
            lead_user_id = inclusion[0]
            if lead_user_id == user.uid:
                continue
            lead_user = users_dict[0]
            top_lead_users.append(lead_user)
        
        offers = calculate_score_for_user(user, top_lead_users)
        result = []
        for offer in offers:
            result.append((user.uid, offer.offer_id, offer.score))

        return result
        


    
    def get_prompt_info_for_user(self, uid: int) -> PromptInfo:
        query = f"""
        WITH 
            cutoff_ts AS (
                SELECT max(timestamp) - 30 * 24 * 3600 * 1000000 as ts_cutoff
                FROM receipts
                WHERE user_id = {uid}
            ),
            payments_agg AS (
                SELECT 
                    user_id,
                    sum(price) AS total_spent_month,
                    count(*) AS transactions_count_month,
                    groupArray((brand_id, price)) AS brand_spendings
                FROM payments
                WHERE user_id = {uid} 
                AND timestamp >= (SELECT ts_cutoff FROM cutoff_ts)
                GROUP BY user_id
            ),
            receipts_agg AS (
                SELECT 
                    r.user_id,
                    groupArray(i.category) AS categories_list,
                    groupArray(i.subcategory) AS subcategories_list
                FROM receipts r
                LEFT JOIN items i ON r.approximate_item_id = i.item_id
                WHERE r.user_id = {uid} 
                AND r.timestamp >= (SELECT ts_cutoff FROM cutoff_ts)
                GROUP BY r.user_id
            )
        SELECT 
            u.user_id,
            u.socdem_cluster,
            u.region,
            COALESCE(p.total_spent_month, 0),
            COALESCE(p.transactions_count_month, 0),--    p.brand_spendings,
            r.categories_list,
            r.subcategories_list
        FROM users u
        LEFT JOIN payments_agg p ON u.user_id = p.user_id
        LEFT JOIN receipts_agg r ON u.user_id = r.user_id
        WHERE u.user_id = {uid};
        """
        user_info_row = self.ch_client.query(query).result_rows[0]
        return PromptInfo(user_info_row[0], user_info_row[1], user_info_row[2], user_info_row[3], user_info_row[5], user_info_row[6])

        


class MockMLService:
    def __init__(self):
        self.is_loaded = False

    def load_resources(self):
        logger.info("ML: Loading FAISS index and Encoder models...")
        # Simulate loading time
        time.sleep(1)
        self.is_loaded = True
        logger.info("ML: Resources loaded.")

    def process_batch(
        self, user_rows: List[Tuple]
    ) -> List[Tuple[int, int, float]]:
        """
        Dummy logic:
        For every user, recommend offer_id=100 with a high score.
        Replace this with: Embeddings Gen -> FAISS Search.
        """
        results = []
        for row in user_rows:
            # Assuming row[0] is user_id based on `SELECT *` or specific query
            user_id = row[0]

            # Simulated inference logic
            recommended_offer_id = 100
            score = 0.95

            results.append((user_id, recommended_offer_id, score))
        return results


class BatchWorker:
    def __init__(self, ml_service: InferenceModel):
        self.ml_service = ml_service
        self.ch_client = None

    def connect_db(self):
        """Establish ClickHouse connection with retry logic."""
        while not self.ch_client:
            try:
                self.ch_client = clickhouse_connect.get_client(
                    host=CH_HOST, port=CH_PORT
                )
                # Verify connection
                self.ch_client.query("SELECT 1")
                logger.info(f"Connected to ClickHouse at {CH_HOST}:{CH_PORT}")
            except Exception as e:
                logger.error(
                    f"Failed to connect to ClickHouse: {e}. Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)

    def get_next_task(self) -> Dict[str, Any]:
        """Polls the coordinator for the next batch."""
        try:
            resp = requests.get(f"{COORDINATOR_URL}/get_task", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 400:
                # Coordinator likely not initialized yet
                logger.warning("Coordinator not initialized. Waiting...")
                return None
            else:
                logger.error(
                    f"Coordinator returned {resp.status_code}: {resp.text}"
                )
                return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Coordinator unavailable: {e}")
            return None

    def fetch_user_data(self, start_id: int, end_id: int) -> List[Tuple]:
        """Fetch raw user data from ClickHouse."""
        query = f"""
            SELECT user_id, socdem_cluster, region 
            FROM users 
            WHERE user_id >= {start_id} AND user_id < {end_id}
        """
        return self.ch_client.query(query).result_rows

    def save_results(self, recommendations: List[Tuple[int, int, float]]):
        """Save predictions back to ClickHouse."""
        if not recommendations:
            return

        # Ensure the table exists (Ideally handled by migration scripts, but safety check here)
        # Table schema assumption: user_id UInt64, offer_id UInt64, score Float32, created_at DateTime

        # Prepare data with timestamp if needed, or just insert as returned by ML
        # Let's add a timestamp for the insert
        data_to_insert = [(*r, int(time.time())) for r in recommendations]

        self.ch_client.insert(
            "recommendations",
            data_to_insert,
            column_names=["user_id", "offer_id", "score", "created_at"],
        )

    def run(self):
        self.connect_db()
        self.ml_service.load_resources()

        logger.info("Worker started. Polling for tasks...")

        while True:
            task = self.get_next_task()

            if not task:
                time.sleep(POLL_INTERVAL)
                continue

            if task["status"] == "empty":
                logger.info(
                    "Queue is empty. Worker going to sleep (or exit)."
                )
                time.sleep(
                    10
                )  # Wait longer before checking again in case of reset
                continue

            batch_id = task["batch_id"]
            start_id = task["start_id"]
            end_id = task["end_id"]

            logger.info(
                f"Processing Batch {batch_id}: User IDs {start_id} to {end_id}"
            )

            try:
                # 1. Fetch Data
                user_rows = self.fetch_user_data(start_id, end_id)
                if not user_rows:
                    logger.info(
                        f"Batch {batch_id} returned no users. Skipping."
                    )
                    continue

                # 2. ML Inference (Protocol usage)
                recommendations = self.ml_service.process_batch(user_rows)

                # 3. Save Results
                self.save_results(recommendations)

                logger.info(
                    f"Batch {batch_id} complete. Processed {len(user_rows)} users."
                )

            except Exception as e:
                logger.error(
                    f"Error processing batch {batch_id}: {e}", exc_info=True
                )
                # In a real scenario, you might want to report failure to Coordinator
                # or rely on a timeout mechanism to re-queue the task.
                time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    historical_service = HistoricalModel()

    worker = BatchWorker(historical_service)
    worker.run()
