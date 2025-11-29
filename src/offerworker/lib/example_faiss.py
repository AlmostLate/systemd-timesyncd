import numpy as np
from collections import defaultdict
import heapq
import time
from typing import Dict, List, Set, Tuple
import faiss
from normalized_faiss import NormalizedFAISS

def generate_synthetic_data(num_records=10000, max_number=200, seed=41):
    """Генерирует синтетические данные"""
    np.random.seed(seed)
    records = {}
    
    # Распределение размеров
    size_distribution = (
        [np.random.randint(1, 6) for _ in range(int(num_records * 0.4))] +
        [np.random.randint(6, 16) for _ in range(int(num_records * 0.3))] +
        [np.random.randint(16, 31) for _ in range(int(num_records * 0.2))] +
        [np.random.randint(31, 51) for _ in range(int(num_records * 0.1))]
    )
    
    # Популярные числа
    popular_numbers = np.random.choice(range(1, max_number + 1), size=20, replace=False)
    
    for record_id in range(num_records):
        if record_id % 100_000 == 0:
            print(record_id)
        if record_id >= len(size_distribution):
            size = np.random.randint(1, 51)
        else:
            size = size_distribution[record_id]
        
        # 70% случайные, 30% популярные
        num_popular = int(size * 0.3)
        num_random = size - num_popular
        
        numbers = list(np.random.choice(popular_numbers, size=num_popular, replace=False))
        
        remaining = list(set(range(1, max_number + 1)) - set(numbers) - set(popular_numbers))
        if len(remaining) >= num_random:
            numbers.extend(np.random.choice(remaining, size=num_random, replace=False))
        
        records[record_id] = list(set(numbers))
    
    return records

records = generate_synthetic_data(num_records=1_000_000)
print(list(records.items())[:10])
norm_faiss = NormalizedFAISS(200)
norm_faiss.build(records)

inclusions = norm_faiss.find_top_inclusions(0, 20, 1000) # list of offerd_ids
print(len(inclusions))
print(inclusions[:-10])
