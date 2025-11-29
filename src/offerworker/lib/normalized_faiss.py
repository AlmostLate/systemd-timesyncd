import numpy as np
from collections import defaultdict
import heapq
import time
from typing import Dict, List, Set, Tuple
import faiss

class NormalizedFAISS:
    """FAISS с L2-нормализацией"""
    
    def __init__(self, offers_count):
        self.offers_count = offers_count
        self.index = None
        self.records = {}
        self.record_id_to_idx = {}
        self.idx_to_record_id = {}
        
    def build(self, records: Dict[int, List[int]], nlist=100):
        print("Building normalized FAISS index...")
        start = time.time()

        self.records = {rid: set(nums) for rid, nums in records.items()}
        
        vectors = []
        idx = 0
        for record_id, numbers in records.items():
            vec = np.zeros(self.offers_count, dtype=np.float32)
            for num in numbers:
                vec[num - 1] = 1.0
            
            vectors.append(vec)
            self.record_id_to_idx[record_id] = idx
            self.idx_to_record_id[idx] = record_id
            idx += 1
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # L2 нормализация
        faiss.normalize_L2(vectors)

        d = vectors.shape[1]
        
        # Inner Product для нормализованных векторов = косинусное сходство
        if len(records) < 1000:
            self.index = faiss.IndexFlatIP(d)
        else:
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, min(nlist, len(records) // 10))
            self.index.train(vectors)
            self.index.nprobe = 10
        
        self.index.add(vectors)
        print(f"  Built in {time.time() - start:.2f}s (indexed {len(records)} records)")
        
    def find_top_inclusions(self, record_id: int, k: int = 100,
                           num_candidates: int = 100) -> List[Tuple[int, float]]:
        print("Querying normalized FAISS index...")
        start = time.time()

        query_set = self.records[record_id]
        query_size = len(query_set)
        
        if query_size == 0:
            return []
        
        if record_id not in self.record_id_to_idx:
            return []
        
        query_vec = np.zeros((1, self.offers_count), dtype=np.float32)
        for num in query_set:
            query_vec[0, num - 1] = 1.0
        
        faiss.normalize_L2(query_vec)
        
        num_to_search = min(num_candidates + 1, len(self.records))
        distances, indices = self.index.search(query_vec, num_to_search)
        
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            
            cand_record_id = self.idx_to_record_id[idx]
            
            if cand_record_id != record_id:
                target_set = self.records[cand_record_id]
                inclusion = len(query_set & target_set) / query_size
                results.append((cand_record_id, inclusion))
        
        print(f"  Queried in {time.time() - start:.2f}s")
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
