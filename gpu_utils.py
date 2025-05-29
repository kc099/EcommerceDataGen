# gpu_utils.py
"""
GPU acceleration utilities using CuPy for high-performance random number generation
and array operations.
"""

import gc
from typing import List, Optional, Union, Any
import numpy as np

# GPU imports with fallback
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration available")
except ImportError:
    print("CuPy not available - falling back to CPU (NumPy)")
    GPU_AVAILABLE = False
    cp = np  # Fallback


class GPUDataGenerator:
    """GPU-accelerated data generation utilities"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            try:
                # Initialize GPU and get device info
                device = cp.cuda.Device()
                # Modern CuPy: use memory pool to get total memory
                mempool = cp.get_default_memory_pool()
                total_mem = mempool.total_bytes()
                print(f"GPU Device: {device}")
                print(f"GPU Memory: {total_mem / 1e9:.1f} GB total")
            except Exception as e:
                print(f"GPU initialization warning: {e}")
                self.use_gpu = False
                self.xp = np
    
    def to_numpy(self, array) -> np.ndarray:
        """Convert GPU array to numpy array"""
        if self.use_gpu and hasattr(array, 'get'):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def random_choice_vectorized(self, choices: List[Any], size: int, 
                               weights: Optional[List[float]] = None,
                               replace: bool = True) -> List[Any]:
        """GPU-accelerated random choice with optional weights. Falls back to numpy for choice due to CuPy limitation."""
        if not choices:
            return []
        # CuPy does not implement random.choice, so always use numpy for this operation
        if weights is not None:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
        indices = np.random.choice(
            len(choices), 
            size=size, 
            p=weights if weights is not None else None,
            replace=replace
        )
        return [choices[i] for i in indices]
    
    def random_uniform_vectorized(self, low: float, high: float, size: int) -> np.ndarray:
        """GPU-accelerated uniform random number generation"""
        result = self.xp.random.uniform(low, high, size=size)
        return self.to_numpy(result)
    
    def random_int_vectorized(self, low: int, high: int, size: int) -> np.ndarray:
        """GPU-accelerated integer random number generation"""
        result = self.xp.random.randint(low, high + 1, size=size)
        return self.to_numpy(result)
    
    def random_normal_vectorized(self, mean: float, std: float, size: int) -> np.ndarray:
        """GPU-accelerated normal distribution"""
        result = self.xp.random.normal(mean, std, size=size)
        return self.to_numpy(result)
    
    def random_exponential_vectorized(self, scale: float, size: int) -> np.ndarray:
        """GPU-accelerated exponential distribution for modeling time between events"""
        result = self.xp.random.exponential(scale, size=size)
        return self.to_numpy(result)
    
    def random_beta_vectorized(self, a: float, b: float, size: int) -> np.ndarray:
        """GPU-accelerated beta distribution for conversion rates"""
        result = self.xp.random.beta(a, b, size=size)
        return self.to_numpy(result)
    
    def weighted_choice_matrix(self, choices_matrix: np.ndarray, 
                             weights_matrix: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated weighted choice for matrices
        Useful for customer behavior modeling with varying preferences
        """
        if self.use_gpu:
            choices_gpu = cp.asarray(choices_matrix)
            weights_gpu = cp.asarray(weights_matrix)
            
            # Normalize weights
            weights_gpu = weights_gpu / weights_gpu.sum(axis=1, keepdims=True)
            
            # Generate cumulative probabilities
            cum_weights = cp.cumsum(weights_gpu, axis=1)
            
            # Generate random numbers
            randoms = cp.random.random(size=(choices_matrix.shape[0], 1))
            
            # Find indices where random < cum_weight
            indices = cp.argmax(randoms < cum_weights, axis=1)
            
            # Get selected choices
            row_indices = cp.arange(choices_matrix.shape[0])
            result = choices_gpu[row_indices, indices]
            
            return cp.asnumpy(result)
        else:
            # NumPy fallback
            results = []
            for i in range(choices_matrix.shape[0]):
                weights = weights_matrix[i] / weights_matrix[i].sum()
                choice_idx = np.random.choice(len(choices_matrix[i]), p=weights)
                results.append(choices_matrix[i, choice_idx])
            return np.array(results)
    
    def generate_customer_segments(self, n_customers: int, 
                                 segment_weights: List[float]) -> np.ndarray:
        """Generate customer segments using GPU acceleration"""
        segment_indices = self.xp.random.choice(
            len(segment_weights), 
            size=n_customers, 
            p=self.xp.array(segment_weights) / sum(segment_weights)
        )
        return self.to_numpy(segment_indices)
    
    def simulate_price_sensitivity(self, base_prices: np.ndarray, 
                                 sensitivity_factors: np.ndarray) -> np.ndarray:
        """
        Simulate price sensitivity effects on conversion probability
        """
        if self.use_gpu:
            base_gpu = cp.asarray(base_prices)
            sensitivity_gpu = cp.asarray(sensitivity_factors)
            
            # Price elasticity simulation
            # Higher sensitivity = lower conversion at higher prices
            normalized_prices = base_gpu / cp.mean(base_gpu)
            price_penalty = cp.power(normalized_prices, sensitivity_gpu)
            
            return cp.asnumpy(1.0 / price_penalty)
        else:
            normalized_prices = base_prices / np.mean(base_prices)
            price_penalty = np.power(normalized_prices, sensitivity_factors)
            return 1.0 / price_penalty
    
    def batch_process(self, total_size: int, batch_size: int, 
                     process_func, *args, **kwargs):
        """
        Process large datasets in batches to manage memory
        """
        results = []
        
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            batch_size_actual = end_idx - start_idx
            
            # Process batch
            batch_result = process_func(batch_size_actual, *args, **kwargs)
            results.append(batch_result)
            
            # Periodic memory cleanup
            if start_idx % (batch_size * 5) == 0:
                self.cleanup_memory()
        
        return results
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            try:
                # Force garbage collection on GPU
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"GPU memory cleanup warning: {e}")
        
        # CPU garbage collection
        gc.collect()
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information"""
        info = {
            "gpu_available": self.use_gpu,
            "using_gpu": self.use_gpu,
        }
        
        if self.use_gpu:
            try:
                mempool = cp.get_default_memory_pool()
                info.update({
                    "gpu_used_bytes": mempool.used_bytes(),
                    "gpu_total_bytes": mempool.total_bytes(),
                    "gpu_usage_percent": (mempool.used_bytes() / mempool.total_bytes() * 100) 
                                        if mempool.total_bytes() > 0 else 0
                })
            except Exception as e:
                info["gpu_error"] = str(e)
        
        return info
    
    def optimize_for_dataset_size(self, dataset_size: int) -> dict:
        """
        Optimize batch sizes and memory settings based on dataset size
        """
        config = {
            "batch_size": 10_000,
            "memory_limit_ratio": 0.8,
        }
        
        if dataset_size < 10_000:
            config["batch_size"] = dataset_size
        elif dataset_size < 100_000:
            config["batch_size"] = 25_000
        elif dataset_size < 1_000_000:
            config["batch_size"] = 50_000
        else:
            config["batch_size"] = 100_000
        
        if self.use_gpu:
            try:
                mempool = cp.get_default_memory_pool()
                total_memory = mempool.total_bytes()
                if total_memory > 0:
                    # Adjust batch size based on available GPU memory
                    available_memory = total_memory * config["memory_limit_ratio"]
                    # Rough estimate: 1M records ≈ 100MB
                    max_batch_from_memory = int(available_memory / (100 * 1024 * 1024))
                    config["batch_size"] = min(config["batch_size"], max_batch_from_memory)
            except Exception:
                pass
        
        return config


# Utility functions for common patterns
def ensure_gpu_array(array, gpu_gen: GPUDataGenerator):
    """Ensure array is on appropriate device (GPU/CPU)"""
    if gpu_gen.use_gpu and not hasattr(array, 'device'):
        return cp.asarray(array)
    return array


def safe_gpu_operation(operation, *args, fallback_operation=None, **kwargs):
    """
    Safely execute GPU operation with automatic fallback
    """
    try:
        if GPU_AVAILABLE:
            return operation(*args, **kwargs)
        elif fallback_operation:
            return fallback_operation(*args, **kwargs)
        else:
            raise NotImplementedError("No fallback operation provided")
    except Exception as e:
        print(f"GPU operation failed: {e}")
        if fallback_operation:
            print("Falling back to CPU operation")
            return fallback_operation(*args, **kwargs)
        else:
            raise