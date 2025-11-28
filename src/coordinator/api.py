import os
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import clickhouse_connect

app = FastAPI(title="Batch Processing Coordinator")

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_PORT = os.getenv("CH_PORT", "8123")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 15000))

task_queue: List[Tuple[int, int]] = []
is_initialized = False


class TaskResponse(BaseModel):
    batch_id: int
    start_id: int
    end_id: int
    status: str  # 'ok' or 'empty'


@app.post("/initialize")
def initialize_tasks():
    """
    1. Идет в БД
    2. Определяет min/max user_id
    3. Нарезает диапазоны и кладет в очередь
    """
    global task_queue, is_initialized

    if is_initialized:
        return {
            "status": "Already initialized",
            "tasks_count": len(task_queue),
        }

    try:
        client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)
        result = client.query(
            "SELECT min(user_id), max(user_id) FROM users"
        ).result_rows
        min_id, max_id = result[0]

        if min_id is None or max_id is None:
            raise HTTPException(
                status_code=400, detail="Users table is empty"
            )

        # Генерация чанков
        current_start = min_id
        while current_start <= max_id:
            current_end = min(current_start + BATCH_SIZE, max_id + 1)
            task_queue.append((current_start, current_end))
            current_start = current_end

        # Разворачиваем, чтобы pop() забирал с начала (хотя порядок не важен)
        task_queue.reverse()
        is_initialized = True

        return {
            "status": "Initialized",
            "min_id": min_id,
            "max_id": max_id,
            "total_batches": len(task_queue),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_task", response_model=TaskResponse)
def get_task():
    """
    Метод для воркеров. Отдает следующий диапазон для обработки.
    """
    global task_queue

    if not is_initialized:
        raise HTTPException(
            status_code=400,
            detail="Coordinator not initialized. Call /initialize first.",
        )

    if not task_queue:
        return TaskResponse(batch_id=-1, start_id=0, end_id=0, status="empty")

    start, end = task_queue.pop()

    return TaskResponse(
        batch_id=len(task_queue),
        start_id=start,
        end_id=end,
        status="ok",
    )


@app.get("/health")
def health():
    return {"queue_size": len(task_queue)}
