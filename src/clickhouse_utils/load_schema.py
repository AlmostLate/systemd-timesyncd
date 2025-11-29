import os
from typing import Protocol
import clickhouse_connect
import time


class Commander(Protocol):
    def command(self, query: str): ...


class Client:
    def __init__(self, base_path: str, commander: Commander):
        self._client = commander
        self._base_path = base_path

    def run_import(self, table_name, file_pattern, source_label=None):
        start_time = time.time()
        full_path = f"{self._base_path}/{file_pattern}"

        # form the select part with additional lable
        if source_label:
            select_clause = f"*, '{source_label}'"
        else:
            select_clause = "*"

        query = f"""
        INSERT INTO {table_name}
        SELECT {select_clause}
        FROM file('{full_path}', 'Parquet')
        """

        try:
            self._client.command(query)
            elapsed = time.time() - start_time
            print(f"    success file {full_path} -- {elapsed:.2f} sec")
        except Exception as e:
            print(f"    Err {file_pattern}: {e}")


def main():
    client = Client(
        base_path=os.environ.get("CH_DATASET_BASE"),
        commander=clickhouse_connect.get_client(
            host=os.environ.get("CH_HOST"),
            port=os.environ.get("CH_PORT"),
            username=os.environ.get("CH_USERNAME"),
            password=os.environ.get("CH_PASSWORD"),
        ),
    )

    client.run_import("users", "users.pq")
    client.run_import("brands", "brands.pq")

    client.run_import("items", "retail/items.pq", source_label="retail")
    client.run_import(
        "items", "marketplace/items.pq", source_label="marketplace"
    )

    client.run_import("events", "retail/events/*.pq", source_label="retail")
    client.run_import(
        "events",
        "marketplace/events/*.pq",
        source_label="marketplace",
    )

    client.run_import("payments", "payments/events/*.pq")
    client.run_import("receipts", "payments/receipts/*.pq")


if __name__ == "__main__":
    main()
