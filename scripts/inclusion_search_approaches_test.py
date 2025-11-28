import numpy as np
from collections import defaultdict
import heapq
import time
from typing import Dict, List, Set, Tuple
import faiss

# ==============================================================================
# 1. ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ
# ==============================================================================

def generate_synthetic_data(num_records=10000, max_number=200, seed=42):
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

# ==============================================================================
# 2. БАЗОВЫЙ INVERTED INDEX
# ==============================================================================

class InvertedIndex:
    """Классический инвертированный индекс (baseline)"""
    
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.records = {}
        
    def build(self, records: Dict[int, List[int]]):
        print("Building inverted index...")
        start = time.time()
        
        for record_id, numbers in records.items():
            self.records[record_id] = set(numbers)
            for num in numbers:
                self.inverted_index[num].append(record_id)
        
        print(f"  Built in {time.time() - start:.2f}s")
        
    def find_top_inclusions(self, record_id: int, k: int = 20) -> List[Tuple[int, float]]:
        query_set = self.records[record_id]
        query_size = len(query_set)
        
        if query_size == 0:
            return []
        
        intersection_counts = defaultdict(int)
        
        for num in query_set:
            for target_id in self.inverted_index[num]:
                if target_id != record_id:
                    intersection_counts[target_id] += 1
        
        heap = []
        for target_id, count in intersection_counts.items():
            percentage = count / query_size
            
            if len(heap) < k:
                heapq.heappush(heap, (percentage, target_id))
            elif percentage > heap[0][0]:
                heapq.heapreplace(heap, (percentage, target_id))
        
        return sorted([(tid, pct) for pct, tid in heap], 
                     key=lambda x: x[1], reverse=True)

# ==============================================================================
# 3. FAISS-BASED ИНДЕКСЫ
# ==============================================================================

class SimpleFAISS:
    """Простой FAISS индекс без нормализации"""
    
    def __init__(self, metric='euclidean'):
        self.metric = metric
        self.index = None
        self.records = {}
        self.record_id_to_idx = {}
        self.idx_to_record_id = {}
        
    def build(self, records: Dict[int, List[int]], nlist=100):
        print(f"Building FAISS index ({self.metric})...")
        start = time.time()
        
        self.records = {rid: set(nums) for rid, nums in records.items()}
        
        # Собираем векторы
        vectors = []
        idx = 0
        for record_id, numbers in records.items():
            vec = np.zeros(200, dtype=np.float32)
            for num in numbers:
                vec[num - 1] = 1.0
            
            vectors.append(vec)
            self.record_id_to_idx[record_id] = idx
            self.idx_to_record_id[idx] = record_id
            idx += 1
        
        vectors = np.array(vectors, dtype=np.float32)
        d = vectors.shape[1]
        
        # Создаём индекс
        if len(records) < 1000:
            # Для маленьких данных - просто Flat индекс
            if self.metric == 'euclidean':
                self.index = faiss.IndexFlatL2(d)
            else:  # angular/cosine
                self.index = faiss.IndexFlatIP(d)
                faiss.normalize_L2(vectors)
        else:
            # Для больших данных - IVF индекс
            quantizer = faiss.IndexFlatL2(d) if self.metric == 'euclidean' else faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, min(nlist, len(records) // 10))
            
            if self.metric != 'euclidean':
                faiss.normalize_L2(vectors)
            
            self.index.train(vectors)
            self.index.nprobe = 10
        
        self.index.add(vectors)
        print(f"  Built in {time.time() - start:.2f}s (indexed {len(records)} records)")
        
    def find_top_inclusions(self, record_id: int, k: int = 20, 
                           num_candidates: int = 100) -> List[Tuple[int, float]]:
        query_set = self.records[record_id]
        query_size = len(query_set)
        
        if query_size == 0:
            return []
        
        if record_id not in self.record_id_to_idx:
            return []
        
        # Создаём вектор запроса
        query_vec = np.zeros((1, 200), dtype=np.float32)
        for num in query_set:
            query_vec[0, num - 1] = 1.0
        
        if self.metric != 'euclidean':
            faiss.normalize_L2(query_vec)
        
        # Ищем соседей
        num_to_search = min(num_candidates + 1, len(self.records))
        distances, indices = self.index.search(query_vec, num_to_search)
        
        # Вычисляем inclusion
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            
            cand_record_id = self.idx_to_record_id[idx]
            
            if cand_record_id != record_id:
                target_set = self.records[cand_record_id]
                inclusion = len(query_set & target_set) / query_size
                results.append((cand_record_id, inclusion))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]


class NormalizedFAISS:
    """FAISS с L2-нормализацией"""
    
    def __init__(self):
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
            vec = np.zeros(200, dtype=np.float32)
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
        
    def find_top_inclusions(self, record_id: int, k: int = 20,
                           num_candidates: int = 100) -> List[Tuple[int, float]]:
        query_set = self.records[record_id]
        query_size = len(query_set)
        
        if query_size == 0:
            return []
        
        if record_id not in self.record_id_to_idx:
            return []
        
        query_vec = np.zeros((1, 200), dtype=np.float32)
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
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]


class SizeStratifiedFAISS:
    """FAISS с группировкой по размеру"""
    
    def __init__(self, bucket_size=10):
        self.bucket_size = bucket_size
        self.indices = {}
        self.records = {}
        
    def build(self, records: Dict[int, List[int]], nlist=50):
        print(f"Building size-stratified FAISS index (bucket_size={self.bucket_size})...")
        start = time.time()
        
        self.records = {rid: set(nums) for rid, nums in records.items()}
        
        # Группируем по размеру
        buckets = defaultdict(lambda: {
            'records': {},
            'vectors': [],
            'record_ids': []
        })
        
        for record_id, numbers in records.items():
            size = len(numbers)
            bucket = size // self.bucket_size
            
            vec = np.zeros(200, dtype=np.float32)
            for num in numbers:
                vec[num - 1] = 1.0
            
            buckets[bucket]['records'][record_id] = set(numbers)
            buckets[bucket]['vectors'].append(vec)
            buckets[bucket]['record_ids'].append(record_id)
        
        # Строим индекс для каждой группы
        for bucket, data in buckets.items():
            vectors = np.array(data['vectors'], dtype=np.float32)
            d = vectors.shape[1]
            
            if len(vectors) < 100:
                index = faiss.IndexFlatL2(d)
            else:
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, min(nlist, len(vectors) // 10))
                index.train(vectors)
                index.nprobe = 5
            
            index.add(vectors)
            data['index'] = index
            
            # Создаём маппинг idx -> record_id
            data['idx_to_record_id'] = {i: rid for i, rid in enumerate(data['record_ids'])}
            data['record_id_to_idx'] = {rid: i for i, rid in enumerate(data['record_ids'])}
            
            self.indices[bucket] = data
        
        print(f"  Built {len(self.indices)} bucket indices in {time.time() - start:.2f}s")
        
    def find_top_inclusions(self, record_id: int, k: int = 20,
                           num_candidates: int = 50) -> List[Tuple[int, float]]:
        query_set = self.records[record_id]
        query_size = len(query_set)
        
        if query_size == 0:
            return []
        
        query_bucket = query_size // self.bucket_size
        
        # Создаём вектор запроса
        query_vec = np.zeros((1, 200), dtype=np.float32)
        for num in query_set:
            query_vec[0, num - 1] = 1.0
        
        all_results = []
        
        # Ищем в своей группе и всех больших
        for bucket in range(query_bucket, max(self.indices.keys()) + 1):
            if bucket not in self.indices:
                continue
            
            bucket_data = self.indices[bucket]
            index = bucket_data['index']
            
            num_to_search = min(num_candidates, len(bucket_data['records']))
            distances, indices = index.search(query_vec, num_to_search)
            
            for idx in indices[0]:
                if idx == -1:
                    continue
                
                cand_record_id = bucket_data['idx_to_record_id'][idx]
                
                if cand_record_id != record_id:
                    target_set = self.records[cand_record_id]
                    inclusion = len(query_set & target_set) / query_size
                    all_results.append((cand_record_id, inclusion))
        
        # Убираем дубликаты
        seen = set()
        unique_results = []
        for rid, score in all_results:
            if rid not in seen:
                seen.add(rid)
                unique_results.append((rid, score))
        
        return sorted(unique_results, key=lambda x: x[1], reverse=True)[:k]


class HybridFAISS:
    """Гибрид FAISS + Inverted Index"""
    
    def __init__(self):
        self.faiss_index = None
        self.inverted_index = defaultdict(list)
        self.records = {}
        self.record_id_to_idx = {}
        self.idx_to_record_id = {}
        
    def build(self, records: Dict[int, List[int]], nlist=100):
        print("Building hybrid FAISS + Inverted index...")
        start = time.time()
        
        self.records = {rid: set(nums) for rid, nums in records.items()}
        
        # Строим FAISS индекс
        vectors = []
        idx = 0
        for record_id, numbers in records.items():
            vec = np.zeros(200, dtype=np.float32)
            for num in numbers:
                vec[num - 1] = 1.0
                # Также строим inverted index
                self.inverted_index[num].append(record_id)
            
            vectors.append(vec)
            self.record_id_to_idx[record_id] = idx
            self.idx_to_record_id[idx] = record_id
            idx += 1
        
        vectors = np.array(vectors, dtype=np.float32)
        d = vectors.shape[1]
        
        if len(records) < 1000:
            self.faiss_index = faiss.IndexFlatL2(d)
        else:
            quantizer = faiss.IndexFlatL2(d)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, d, min(nlist, len(records) // 10))
            self.faiss_index.train(vectors)
            self.faiss_index.nprobe = 10
        
        self.faiss_index.add(vectors)
        print(f"  Built in {time.time() - start:.2f}s")
        
    def find_top_inclusions(self, record_id: int, k: int = 20,
                           num_faiss_candidates: int = 200,
                           num_inv_candidates: int = 1000) -> List[Tuple[int, float]]:
        query_set = self.records[record_id]
        query_size = len(query_set)
        
        if query_size == 0:
            return []
        
        # FAISS кандидаты
        query_vec = np.zeros((1, 200), dtype=np.float32)
        for num in query_set:
            query_vec[0, num - 1] = 1.0
        
        num_to_search = min(num_faiss_candidates + 1, len(self.records))
        distances, indices = self.faiss_index.search(query_vec, num_to_search)
        
        faiss_candidates = set()
        for idx in indices[0]:
            if idx != -1:
                faiss_candidates.add(self.idx_to_record_id[idx])
        
        # Inverted index кандидаты
        inv_candidates = set()
        for num in query_set:
            inv_candidates.update(self.inverted_index[num][:num_inv_candidates])
        
        # Объединяем
        all_candidates = faiss_candidates | inv_candidates
        
        # Точный расчёт
        results = []
        for cid in all_candidates:
            if cid != record_id:
                target_set = self.records[cid]
                inclusion = len(query_set & target_set) / query_size
                results.append((cid, inclusion))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

# ==============================================================================
# 4. ТЕСТИРОВАНИЕ
# ==============================================================================

def compute_ground_truth(records: Dict[int, Set[int]], 
                        query_id: int, 
                        k: int = 20) -> List[Tuple[int, float]]:
    """Вычисляет истинный топ-K"""
    query_set = records[query_id]
    query_size = len(query_set)
    
    if query_size == 0:
        return []
    
    results = []
    for target_id, target_set in records.items():
        if target_id != query_id:
            inclusion = len(query_set & target_set) / query_size
            if inclusion > 0:
                results.append((target_id, inclusion))
    
    return sorted(results, key=lambda x: x[1], reverse=True)[:k]


def evaluate_method(method, records: Dict[int, List[int]], 
                   num_queries: int = 100, k: int = 20, verbose=False):
    """Оценивает качество метода"""
    
    records_sets = {rid: set(nums) for rid, nums in records.items()}
    
    record_ids = list(records.keys())
    np.random.seed(42)
    query_ids = np.random.choice(record_ids, size=min(num_queries, len(record_ids)), 
                                 replace=False)
    
    metrics = {
        'recall@5': [],
        'recall@10': [],
        'recall@20': [],
        'precision@20': [],
        'ndcg@20': [],
        'query_times': []
    }
    
    for i, query_id in enumerate(query_ids):
        gt = compute_ground_truth(records_sets, query_id, k=20)
        
        if len(gt) == 0:
            continue
        
        gt_ids = set(r[0] for r in gt)
        gt_dict = {r[0]: r[1] for r in gt}
        
        # Предсказание метода
        start = time.time()
        try:
            pred = method.find_top_inclusions(query_id, k=20)
        except Exception as e:
            if verbose:
                print(f"Error for query {query_id}: {e}")
            continue
        query_time = time.time() - start
        
        if verbose and i < 3:
            print(f"\nQuery {query_id} (size={len(records_sets[query_id])}):")
            print(f"  GT top-5:   {gt[:5]}")
            print(f"  Pred top-5: {pred[:5]}")
        
        pred_ids = set(r[0] for r in pred[:20])
        
        # Метрики
        for top_k in [5, 10, 20]:
            pred_k = set(r[0] for r in pred[:top_k])
            gt_k = set(r[0] for r in gt[:top_k])
            if len(gt_k) > 0:
                recall = len(pred_k & gt_k) / len(gt_k)
                metrics[f'recall@{top_k}'].append(recall)
        
        if len(pred_ids) > 0:
            precision = len(pred_ids & gt_ids) / len(pred_ids)
            metrics['precision@20'].append(precision)
        
        # NDCG@20
        dcg = 0
        idcg = 0
        for i, (pred_id, _) in enumerate(pred[:20], 1):
            relevance = gt_dict.get(pred_id, 0)
            dcg += relevance / np.log2(i + 1)
        
        for i, (_, score) in enumerate(gt[:20], 1):
            idcg += score / np.log2(i + 1)
        
        if idcg > 0:
            ndcg = dcg / idcg
            metrics['ndcg@20'].append(ndcg)
        
        metrics['query_times'].append(query_time)
    
    # Агрегируем результаты
    results = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'count': len(values)
            }
        else:
            results[metric_name] = {
                'mean': 0,
                'std': 0,
                'median': 0,
                'count': 0
            }
    
    return results


def print_evaluation_results(name: str, results: dict):
    """Красиво печатает результаты"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    for metric, stats in results.items():
        if metric == 'query_times':
            print(f"{metric:20s}: {stats['mean']*1000:6.2f}ms ± {stats['std']*1000:5.2f}ms "
                  f"(median: {stats['median']*1000:6.2f}ms, n={stats['count']})")
        else:
            print(f"{metric:20s}: {stats['mean']*100:5.1f}% ± {stats['std']*100:4.1f}% "
                  f"(n={stats['count']})")


def run_comprehensive_test(num_records=10000, num_queries=100):
    """Запускает полное тестирование всех методов"""
    
    print("="*60)
    print("  COMPREHENSIVE EVALUATION WITH FAISS")
    print("="*60)
    print(f"Records: {num_records:,}")
    print(f"Queries: {num_queries}")
    print("="*60)
    
    # Генерируем данные
    print("\nGenerating synthetic data...")
    records = generate_synthetic_data(num_records=num_records)
    
    # Статистика данных
    sizes = [len(nums) for nums in records.values()]
    print(f"\nData statistics:")
    print(f"  Size min/max/mean: {min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}")
    print(f"  Size median: {np.median(sizes):.1f}")
    print(f"  Size std: {np.std(sizes):.1f}")
    
    # Тестируем методы
    methods = {}
    
    # 1. Inverted Index (baseline)
    print("\n" + "="*60)
    inv_idx = InvertedIndex()
    inv_idx.build(records)
    methods['Inverted Index'] = inv_idx
    
    # 2. Simple FAISS (euclidean)
    print("\n" + "="*60)
    simple_faiss = SimpleFAISS(metric='euclidean')
    simple_faiss.build(records)
    methods['FAISS (binary, euclidean)'] = simple_faiss
    
    # 3. Simple FAISS (angular/cosine)
    print("\n" + "="*60)
    simple_faiss_angular = SimpleFAISS(metric='angular')
    simple_faiss_angular.build(records)
    methods['FAISS (binary, angular)'] = simple_faiss_angular
    
    # 4. Normalized FAISS
    print("\n" + "="*60)
    norm_faiss = NormalizedFAISS()
    norm_faiss.build(records)
    methods['FAISS (normalized)'] = norm_faiss
    
    # 5. Size-stratified FAISS
    print("\n" + "="*60)
    strat_faiss = SizeStratifiedFAISS(bucket_size=10)
    strat_faiss.build(records)
    methods['FAISS (size-stratified)'] = strat_faiss
    
    # 6. Hybrid FAISS
    print("\n" + "="*60)
    hybrid = HybridFAISS()
    hybrid.build(records)
    methods['Hybrid (FAISS + Inverted)'] = hybrid
    
    # Оцениваем каждый метод
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    
    all_results = {}
    for name, method in methods.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_method(method, records, num_queries=num_queries,
                                 verbose=(name == 'FAISS (binary, euclidean)'))
        all_results[name] = results
        print_evaluation_results(name, results)
    
    # Сводная таблица
    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(f"\n{'Method':<30s} {'Recall@20':>12s} {'Time (ms)':>12s}")
    print("-" * 60)
    for name, results in all_results.items():
        recall = results['recall@20']['mean'] * 100
        time_ms = results['query_times']['mean'] * 1000
        print(f"{name:<30s} {recall:11.1f}% {time_ms:11.2f}ms")
    
    return all_results, records, methods


# ==============================================================================
# 5. АНАЛИЗ ПО РАЗМЕРУ ЗАПРОСА
# ==============================================================================

def analyze_by_query_size(methods, records, all_results):
    """Анализирует качество в зависимости от размера запроса"""
    
    print("\n" + "="*60)
    print("  ANALYSIS BY QUERY SIZE")
    print("="*60)
    
    records_sets = {rid: set(nums) for rid, nums in records.items()}
    
    size_bins = [(1, 5), (6, 10), (11, 20), (21, 50)]
    
    for method_name, method in methods.items():
        print(f"\n{method_name}:")
        
        for min_size, max_size in size_bins:
            # Находим запросы в этом диапазоне
            queries_in_bin = [
                rid for rid, nums in records.items()
                if min_size <= len(nums) <= max_size
            ]
            
            if len(queries_in_bin) < 10:
                continue
            
            # Выбираем случайные 20 запросов
            np.random.seed(42)
            sample_queries = np.random.choice(
                queries_in_bin, 
                size=min(20, len(queries_in_bin)), 
                replace=False
            )
            
            recalls = []
            times = []
            
            for query_id in sample_queries:
                gt = compute_ground_truth(records_sets, query_id, k=20)
                if len(gt) == 0:
                    continue
                
                gt_ids = set(r[0] for r in gt)
                
                start = time.time()
                pred = method.find_top_inclusions(query_id, k=20)
                query_time = time.time() - start
                
                pred_ids = set(r[0] for r in pred[:20])
                
                if len(gt_ids) > 0:
                    recall = len(pred_ids & gt_ids) / len(gt_ids)
                    recalls.append(recall)
                    times.append(query_time)
            
            if len(recalls) > 0:
                print(f"  Size [{min_size:2d}-{max_size:2d}]: "
                      f"Recall@20 = {np.mean(recalls)*100:5.1f}% ± {np.std(recalls)*100:4.1f}%, "
                      f"Time = {np.mean(times)*1000:5.2f}ms")


# ==============================================================================
# 6. MAIN
# ==============================================================================

if __name__ == "__main__":
    # Запускаем полное тестирование
    all_results, records, methods = run_comprehensive_test(
        num_records=100000,
        num_queries=1000
    )
    
    # Дополнительный анализ
    analyze_by_query_size(methods, records, all_results)
    
    print("\n" + "="*60)
    print("  TESTING COMPLETE")
    print("="*60)

# ==============================================================================
# 7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(all_results: dict, save_path='method_comparison.png'):
    """Визуализирует результаты сравнения методов"""
    
    # Настройка стиля
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparison of Inclusion Search Methods', fontsize=16, fontweight='bold')
    
    methods = list(all_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # =========================================================================
    # 1. Recall Comparison
    # =========================================================================
    ax = axes[0, 0]
    recall_metrics = ['recall@5', 'recall@10', 'recall@20']
    x = np.arange(len(methods))
    width = 0.25
    
    for i, metric in enumerate(recall_metrics):
        values = [all_results[m][metric]['mean'] * 100 for m in methods]
        stds = [all_results[m][metric]['std'] * 100 for m in methods]
        ax.bar(x + i*width, values, width, label=metric.replace('recall@', 'R@'), 
               yerr=stds, capsize=3, alpha=0.8)
    
    ax.set_ylabel('Recall (%)', fontweight='bold')
    ax.set_title('Recall Comparison', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # =========================================================================
    # 2. Query Time Comparison (log scale)
    # =========================================================================
    ax = axes[0, 1]
    times = [all_results[m]['query_times']['mean'] * 1000 for m in methods]
    stds = [all_results[m]['query_times']['std'] * 1000 for m in methods]
    
    bars = ax.bar(range(len(methods)), times, yerr=stds, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Добавляем значения на столбцы
    for i, (bar, time) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}ms',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Query Time (ms)', fontweight='bold')
    ax.set_title('Query Time Comparison (lower is better)', fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    # =========================================================================
    # 3. NDCG@20 Comparison
    # =========================================================================
    ax = axes[0, 2]
    ndcg_values = [all_results[m]['ndcg@20']['mean'] * 100 for m in methods]
    ndcg_stds = [all_results[m]['ndcg@20']['std'] * 100 for m in methods]
    
    bars = ax.bar(range(len(methods)), ndcg_values, yerr=ndcg_stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Добавляем значения
    for bar, ndcg in zip(bars, ndcg_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ndcg:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('NDCG@20 (%)', fontweight='bold')
    ax.set_title('NDCG@20 Comparison (Ranking Quality)', fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # =========================================================================
    # 4. Precision vs Recall Tradeoff
    # =========================================================================
    ax = axes[1, 0]
    precisions = [all_results[m]['precision@20']['mean'] * 100 for m in methods]
    recalls = [all_results[m]['recall@20']['mean'] * 100 for m in methods]
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.scatter(recalls[i], precisions[i], s=200, color=color, 
                  alpha=0.7, edgecolor='black', linewidth=2, zorder=3)
        
        # Аннотация с небольшим смещением
        offset = 2 if i % 2 == 0 else -2
        ax.annotate(method, (recalls[i], precisions[i]), 
                   xytext=(5, offset), textcoords='offset points', 
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Recall@20 (%)', fontweight='bold')
    ax.set_ylabel('Precision@20 (%)', fontweight='bold')
    ax.set_title('Precision vs Recall Tradeoff', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim([min(recalls)-5, 105])
    ax.set_ylim([min(precisions)-5, 105])
    
    # Диагональная линия (идеальный случай)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
    
    # =========================================================================
    # 5. Speed vs Quality Tradeoff
    # =========================================================================
    ax = axes[1, 1]
    times_log = [all_results[m]['query_times']['mean'] * 1000 for m in methods]
    recalls_quality = [all_results[m]['recall@20']['mean'] * 100 for m in methods]
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.scatter(times_log[i], recalls_quality[i], s=200, color=color,
                  alpha=0.7, edgecolor='black', linewidth=2, zorder=3)
        
        # Аннотация
        offset = 2 if i % 2 == 0 else -2
        ax.annotate(method, (times_log[i], recalls_quality[i]),
                   xytext=(5, offset), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Query Time (ms, log scale)', fontweight='bold')
    ax.set_ylabel('Recall@20 (%)', fontweight='bold')
    ax.set_title('Speed vs Quality Tradeoff', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, which='both')
    
    # Отмечаем "идеальную" зону (быстро и точно)
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax.fill_between([0.1, 10], 95, 100, alpha=0.1, color='green', label='Ideal zone')
    ax.legend(loc='lower left')
    
    # =========================================================================
    # 6. All Metrics Comparison (Radar/Spider Chart)
    # =========================================================================
    ax = axes[1, 2]
    
    # Нормализуем метрики для радара (все от 0 до 100)
    metrics_for_radar = ['recall@20', 'precision@20', 'ndcg@20']
    
    # Количество метрик
    num_metrics = len(metrics_for_radar)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    # Рисуем для каждого метода
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [all_results[method][m]['mean'] * 100 for m in metrics_for_radar]
        values += values[:1]  # Замыкаем круг
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color, alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Настройка осей
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_for_radar], 
                       fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True, alpha=0.3)
    ax.set_title('Overall Performance (Radar Chart)', fontweight='bold', pad=20)
    
    # Легенда вне графика
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    # =========================================================================
    # Сохраняем
    # =========================================================================
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as '{save_path}'")
    plt.show()


def plot_size_analysis(methods, records, save_path='size_analysis.png'):
    """Визуализирует зависимость качества от размера запроса"""
    
    print("\nGenerating size analysis plot...")
    
    records_sets = {rid: set(nums) for rid, nums in records.items()}
    size_bins = [(1, 5), (6, 10), (11, 20), (21, 50)]
    
    # Собираем данные
    results_by_size = {method_name: {'bins': [], 'recalls': [], 'times': []} 
                       for method_name in methods.keys()}
    
    for min_size, max_size in size_bins:
        queries_in_bin = [
            rid for rid, nums in records.items()
            if min_size <= len(nums) <= max_size
        ]
        
        if len(queries_in_bin) < 10:
            continue
        
        np.random.seed(42)
        sample_queries = np.random.choice(
            queries_in_bin, 
            size=min(30, len(queries_in_bin)), 
            replace=False
        )
        
        bin_label = f"{min_size}-{max_size}"
        
        for method_name, method in methods.items():
            recalls = []
            times = []
            
            for query_id in sample_queries:
                gt = compute_ground_truth(records_sets, query_id, k=20)
                if len(gt) == 0:
                    continue
                
                gt_ids = set(r[0] for r in gt)
                
                start = time.time()
                pred = method.find_top_inclusions(query_id, k=20)
                query_time = time.time() - start
                
                pred_ids = set(r[0] for r in pred[:20])
                
                if len(gt_ids) > 0:
                    recall = len(pred_ids & gt_ids) / len(gt_ids)
                    recalls.append(recall)
                    times.append(query_time)
            
            if len(recalls) > 0:
                results_by_size[method_name]['bins'].append(bin_label)
                results_by_size[method_name]['recalls'].append(np.mean(recalls) * 100)
                results_by_size[method_name]['times'].append(np.mean(times) * 1000)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance by Query Size', fontsize=14, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # График 1: Recall по размеру
    ax = axes[0]
    for (method_name, data), color in zip(results_by_size.items(), colors):
        if len(data['bins']) > 0:
            ax.plot(data['bins'], data['recalls'], 'o-', linewidth=2, 
                   markersize=8, label=method_name, color=color, alpha=0.7)
    
    ax.set_xlabel('Query Size Range', fontweight='bold')
    ax.set_ylabel('Recall@20 (%)', fontweight='bold')
    ax.set_title('Recall@20 by Query Size', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    # График 2: Time по размеру
    ax = axes[1]
    for (method_name, data), color in zip(results_by_size.items(), colors):
        if len(data['bins']) > 0:
            ax.plot(data['bins'], data['times'], 'o-', linewidth=2,
                   markersize=8, label=method_name, color=color, alpha=0.7)
    
    ax.set_xlabel('Query Size Range', fontweight='bold')
    ax.set_ylabel('Query Time (ms)', fontweight='bold')
    ax.set_title('Query Time by Query Size', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Size analysis plot saved as '{save_path}'")
    plt.show()


def plot_detailed_comparison_table(all_results, save_path='comparison_table.png'):
    """Создаёт детальную таблицу сравнения"""
    
    print("\nGenerating detailed comparison table...")
    
    methods = list(all_results.keys())
    
    # Подготовка данных для таблицы
    table_data = []
    metrics_display = [
        ('recall@5', 'R@5'),
        ('recall@10', 'R@10'),
        ('recall@20', 'R@20'),
        ('precision@20', 'P@20'),
        ('ndcg@20', 'NDCG@20'),
        ('query_times', 'Time')
    ]
    
    for method in methods:
        row = [method]
        for metric, _ in metrics_display:
            mean = all_results[method][metric]['mean']
            std = all_results[method][metric]['std']
            
            if metric == 'query_times':
                row.append(f"{mean*1000:.2f}±{std*1000:.2f}ms")
            else:
                row.append(f"{mean*100:.1f}±{std*100:.1f}%")
        
        table_data.append(row)
    
    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Заголовки
    headers = ['Method'] + [display for _, display in metrics_display]
    
    # Создаём таблицу
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25] + [0.125]*6)
    
    # Стилизация
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Раскраска заголовков
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Раскраска строк
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for i in range(len(methods)):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors[i])
            cell.set_alpha(0.3)
    
    plt.title('Detailed Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison table saved as '{save_path}'")
    plt.show()


# ==============================================================================
# 8. ОБНОВЛЁННЫЙ MAIN
# ==============================================================================

if __name__ == "__main__":
    # Запускаем полное тестирование
    all_results, records, methods = run_comprehensive_test(
        num_records=10000,
        num_queries=100
    )
    
    # Дополнительный анализ по размеру
    analyze_by_query_size(methods, records, all_results)
    
    # Визуализации
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_results(all_results, save_path='method_comparison.png')
    plot_size_analysis(methods, records, save_path='size_analysis.png')
    plot_detailed_comparison_table(all_results, save_path='comparison_table.png')
    
    print("\n" + "="*60)
    print("  TESTING COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - method_comparison.png")
    print("  - size_analysis.png")
    print("  - comparison_table.png")
