# main.py
"""
Main entry point for the GPU-optimized e-commerce data generator with
advanced inventory management and customer behavior modeling.

Usage:
python main.py --num-customers 50000 --num-products 20000 --use-gpu
"""

import argparse
import gc
from datetime import date, datetime
from pathlib import Path

from config import DEFAULT_CONFIG
from data_generators import EcommerceDataPipeline
from data_validation import DataValidator
from gpu_utils import GPUDataGenerator

VIEWS_PER_CUSTOMER_PER_DAY = 0.5

def main(
    outdir: Path,
    num_customers: int,
    num_products: int,
    num_merchants: int,
    start: date,
    end: date,
    use_gpu: bool = True,
    avg_views_per_day: int = 5000,
    validate_data: bool = True,
):
    """Main data generation pipeline"""
    
    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GPU generator
    gpu_gen = GPUDataGenerator(use_gpu)
    
    print(f"\n{'='*60}")
    print(f"E-COMMERCE DATA GENERATOR - GPU OPTIMIZED")
    print(f"{'='*60}")
    
    if gpu_gen.use_gpu:
        print("✓ GPU acceleration enabled (CuPy)")
    else:
        print("⚠ Using CPU fallback (install CuPy for GPU acceleration)")
    
    print(f"\nDataset Configuration:")
    print(f"  Customers: {num_customers:,}")
    print(f"  Products: {num_products:,}")
    print(f"  Merchants: {num_merchants:,}")
    print(f"  Date range: {start} to {end}")
    print(f"  Avg views/day: {avg_views_per_day:,}")
    print(f"  Output directory: {outdir}")
    
    # Initialize the data pipeline
    pipeline = EcommerceDataPipeline(gpu_gen)
    
    # Generate all datasets
    print(f"\n{'='*40}")
    print("GENERATING DATASETS")
    print(f"{'='*40}")

    avg_views_per_day = int(num_customers * VIEWS_PER_CUSTOMER_PER_DAY)
    
    datasets = pipeline.generate_full_dataset(
        num_customers=num_customers,
        num_products=num_products,
        num_merchants=num_merchants,
        start_date=start,
        end_date=end,
        avg_views_per_day=avg_views_per_day
    )
    
    # Save datasets
    print(f"\n{'='*40}")
    print("SAVING DATASETS")
    print(f"{'='*40}")
    
    for filename, df in datasets.items():
        filepath = outdir / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {filename}: {len(df):,} rows")
    
    # Validate data quality
    if validate_data:
        print(f"\n{'='*40}")
        print("VALIDATING DATA QUALITY")
        print(f"{'='*40}")
        
        validator = DataValidator()
        validation_results = validator.validate_full_dataset(datasets)
        
        if validation_results['is_valid']:
            print("✓ All validation checks passed!")
        else:
            print("⚠ Data validation issues found:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
    
    # Generate summary report
    print(f"\n{'='*40}")
    print("DATASET SUMMARY")
    print(f"{'='*40}")
    
    total_rows = sum(len(df) for df in datasets.values())
    print(f"Total records generated: {total_rows:,}")
    print(f"Total files: {len(datasets)}")
    
    # Memory cleanup
    if gpu_gen.use_gpu:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass
    
    gc.collect()
    print(f"\n✓ Data generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-optimized e-commerce data generator")
    
    # Dataset size parameters
    parser.add_argument("--num-customers", type=int, default=DEFAULT_CONFIG['num_customers'])
    parser.add_argument("--num-products", type=int, default=DEFAULT_CONFIG['num_products'])
    parser.add_argument("--num-merchants", type=int, default=DEFAULT_CONFIG['num_merchants'])
    
    # Date range
    parser.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), 
                       default=DEFAULT_CONFIG['start_date'])
    parser.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), 
                       default=DEFAULT_CONFIG['end_date'])
    
    # Performance parameters
    parser.add_argument("--avg-views-per-day", type=int, default=5000)
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    
    # Output and validation
    parser.add_argument("--outdir", type=Path, default=Path("./ecommerce_data"))
    parser.add_argument("--no-validation", dest="validate_data", action="store_false", default=True)
    
    args = parser.parse_args()
    
    main(
        outdir=args.outdir,
        num_customers=args.num_customers,
        num_products=args.num_products,
        num_merchants=args.num_merchants,
        start=args.start,
        end=args.end,
        use_gpu=args.use_gpu,
        avg_views_per_day=args.avg_views_per_day,
        validate_data=args.validate_data,
    )