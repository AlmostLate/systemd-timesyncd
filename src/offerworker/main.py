import os
import time
import logging
import requests
from typing import List, Tuple, Protocol, Dict, Any
import clickhouse_connect

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
            SELECT user_id, age, gender, income 
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
    ml_service = MockMLService()

    worker = BatchWorker(ml_service)
    worker.run()
