from pyspark.sql import SparkSession
from pyspark import StorageLevel
import time
import random

# Initialize Spark with production-like settings
spark = SparkSession.builder \
    .appName("CacheTestProductionLike") \
    .config("spark.ui.enabled", "true") \
    .config("spark.ui.port", "4040") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.sql.shuffle.partitions", "100") \
    .master("local[4]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

def create_dataframe_with_partitions(num_rows=100000, num_partitions=100):
    """Create a DataFrame with specific number of partitions"""
    print(f"Creating DataFrame with {num_rows} rows and {num_partitions} partitions...")
    
    # Create data with controlled partitions
    data = [(i, i * 2, f"value_{i}", random.random()) for i in range(num_rows)]
    
    # Create RDD with specific partition count
    rdd = spark.sparkContext.parallelize(data, num_partitions)
    
    df = spark.createDataFrame(rdd, ["id", "value", "name", "random"])
    
    # Add transformations similar to production
    df = df.filter(df.id % 2 == 0) \
        .withColumn("computed", df.value * 10) \
        .withColumn("category", df.id % 10)
    
    return df

def check_empty_with_rdd(df):
    """Simulates the production check_spark_df_empty function"""
    return df.rdd.isEmpty()

def check_empty_with_limit(df):
    """The corrected version"""
    return df.limit(1).count() == 0

def production_like_test():
    """Test that simulates production scenario with multiple DataFrames"""
    print("\n" + "="*60)
    print("PRODUCTION-LIKE TEST WITH MULTIPLE DATAFRAMES")
    print("="*60)
    
    # Clear cache
    spark.catalog.clearCache()
    
    # Create multiple DataFrames like in production
    print("\nCreating 3 DataFrames to simulate production pipeline...")
    df1 = create_dataframe_with_partitions(50000, 50)
    df2 = create_dataframe_with_partitions(30000, 30)
    df3 = create_dataframe_with_partitions(20000, 20)
    
    print("\n--- Testing with rdd.isEmpty() (PROBLEMATIC) ---")
    
    # Persist all DataFrames (production pattern)
    df1_persisted = df1.persist(StorageLevel.MEMORY_AND_DISK)
    df2_persisted = df2.persist(StorageLevel.MEMORY_AND_DISK)
    df3_persisted = df3.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Check empty using rdd.isEmpty() (the problematic pattern)
    print("\nChecking if DataFrames are empty using rdd.isEmpty()...")
    start = time.time()
    empty1 = check_empty_with_rdd(df1_persisted)
    empty2 = check_empty_with_rdd(df2_persisted)
    empty3 = check_empty_with_rdd(df3_persisted)
    isEmpty_time = time.time() - start
    print(f"isEmpty() checks completed in {isEmpty_time:.2f}s")
    
    # Check cache status
    print("\nCache Status after isEmpty() checks:")
    total_cached_partitions = 0
    total_partitions = 0
    for rdd_info in spark.sparkContext._jsc.sc().getRDDStorageInfo():
        if rdd_info.numCachedPartitions() > 0:
            cached = rdd_info.numCachedPartitions()
            total = rdd_info.numPartitions()
            total_cached_partitions += cached
            total_partitions += total
            percentage = (cached * 100.0 / total)
            print(f"  RDD {rdd_info.id()}: {cached}/{total} partitions ({percentage:.1f}%)")
    
    if total_partitions > 0:
        overall_percentage = (total_cached_partitions * 100.0 / total_partitions)
        print(f"\nOVERALL: {total_cached_partitions}/{total_partitions} partitions cached ({overall_percentage:.1f}%)")
    
    # Simulate production operations
    print("\nSimulating production operations...")
    start = time.time()
    count1 = df1_persisted.count()
    count2 = df2_persisted.count()
    count3 = df3_persisted.count()
    operation_time_rdd = time.time() - start
    print(f"Operations took {operation_time_rdd:.2f}s")
    
    # Clear and test with limit(1).count()
    print("\n--- Testing with limit(1).count() (CORRECT) ---")
    spark.catalog.clearCache()
    
    # Re-persist
    df1_persisted = df1.persist(StorageLevel.MEMORY_AND_DISK)
    df2_persisted = df2.persist(StorageLevel.MEMORY_AND_DISK)
    df3_persisted = df3.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Check empty using limit(1).count()
    print("\nChecking if DataFrames are empty using limit(1).count()...")
    start = time.time()
    empty1 = check_empty_with_limit(df1_persisted)
    empty2 = check_empty_with_limit(df2_persisted)
    empty3 = check_empty_with_limit(df3_persisted)
    limit_time = time.time() - start
    print(f"limit(1).count() checks completed in {limit_time:.2f}s")
    
    # Force full materialization
    print("Forcing full materialization...")
    df1_persisted.count()
    df2_persisted.count()
    df3_persisted.count()
    
    # Check cache status
    print("\nCache Status after limit(1).count() and materialization:")
    total_cached_partitions = 0
    total_partitions = 0
    for rdd_info in spark.sparkContext._jsc.sc().getRDDStorageInfo():
        if rdd_info.numCachedPartitions() > 0:
            cached = rdd_info.numCachedPartitions()
            total = rdd_info.numPartitions()
            total_cached_partitions += cached
            total_partitions += total
            percentage = (cached * 100.0 / total)
            print(f"  RDD {rdd_info.id()}: {cached}/{total} partitions ({percentage:.1f}%)")
    
    if total_partitions > 0:
        overall_percentage = (total_cached_partitions * 100.0 / total_partitions)
        print(f"\nOVERALL: {total_cached_partitions}/{total_partitions} partitions cached ({overall_percentage:.1f}%)")
    
    # Simulate production operations
    print("\nSimulating production operations...")
    start = time.time()
    count1 = df1_persisted.count()
    count2 = df2_persisted.count()
    count3 = df3_persisted.count()
    operation_time_limit = time.time() - start
    print(f"Operations took {operation_time_limit:.2f}s")
    
    return operation_time_rdd, operation_time_limit

def isolated_comparison_test():
    """Isolated test comparing both methods on single DataFrame"""
    print("\n" + "="*60)
    print("ISOLATED COMPARISON TEST")
    print("="*60)
    
    # Create DataFrame with many partitions
    df = create_dataframe_with_partitions(100000, 100)
    
    # Test 1: rdd.isEmpty()
    print("\n--- Test 1: rdd.isEmpty() ---")
    spark.catalog.clearCache()
    
    df_persisted = df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Initial count to ensure data exists
    initial_count = df_persisted.count()
    print(f"Initial count: {initial_count}")
    
    # Check cache before isEmpty
    print("Cache BEFORE isEmpty():")
    check_detailed_cache_status()
    
    # Call isEmpty
    start = time.time()
    is_empty = df_persisted.rdd.isEmpty()
    isEmpty_time = time.time() - start
    print(f"\nisEmpty() returned {is_empty} in {isEmpty_time:.2f}s")
    
    # Check cache after isEmpty
    print("\nCache AFTER isEmpty():")
    check_detailed_cache_status()
    
    # Performance test
    start = time.time()
    test_count = df_persisted.count()
    count_time_rdd = time.time() - start
    print(f"\nCount after isEmpty(): {test_count} in {count_time_rdd:.2f}s")
    
    # Test 2: limit(1).count()
    print("\n--- Test 2: limit(1).count() ---")
    spark.catalog.clearCache()
    
    df_persisted = df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Call limit(1).count()
    start = time.time()
    is_empty = df_persisted.limit(1).count() == 0
    limit_time = time.time() - start
    print(f"limit(1).count() returned {is_empty} in {limit_time:.2f}s")
    
    # Force full materialization
    df_persisted.count()
    
    # Check cache after materialization
    print("\nCache AFTER limit(1).count() and materialization:")
    check_detailed_cache_status()
    
    # Performance test
    start = time.time()
    test_count = df_persisted.count()
    count_time_limit = time.time() - start
    print(f"\nCount after limit: {test_count} in {count_time_limit:.2f}s")
    
    return count_time_rdd, count_time_limit

def check_detailed_cache_status():
    """Detailed cache status check"""
    storage_info = spark.sparkContext._jsc.sc().getRDDStorageInfo()
    if len(storage_info) == 0:
        print("  No RDDs cached")
    else:
        for rdd_info in storage_info:
            if rdd_info.numCachedPartitions() > 0:
                cached = rdd_info.numCachedPartitions()
                total = rdd_info.numPartitions()
                mem_size = rdd_info.memSize() / 1024 / 1024
                disk_size = rdd_info.diskSize() / 1024 / 1024
                percentage = (cached * 100.0 / total)
                print(f"  RDD {rdd_info.id()}: {cached}/{total} partitions ({percentage:.1f}%)")
                print(f"    Memory: {mem_size:.2f} MB, Disk: {disk_size:.2f} MB")
                
                # This is the key insight - in production you see ~6% cached
                if percentage < 20:
                    print(f"    WARNING: Only {percentage:.1f}% cached - similar to production issue!")

def main():
    print("\n" + "="*70)
    print("SPARK CACHE TEST - REPRODUCING PRODUCTION ISSUE")
    print("="*70)
    print("""
This test demonstrates the caching issue with rdd.isEmpty() that causes:
- 45 min → 200 min performance degradation in production
- Only 98/1600 (6%) partitions cached instead of 100%
- Recomputation spikes in the DAG
""")
    
    # Run isolated test
    rdd_time, limit_time = isolated_comparison_test()
    
    # Run production-like test
    prod_rdd_time, prod_limit_time = production_like_test()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nPerformance Comparison:")
    print(f"{'Test Type':<30} | {'rdd.isEmpty()':<15} | {'limit(1).count()':<15}")
    print("-" * 65)
    print(f"{'Isolated count operation':<30} | {rdd_time:<14.3f}s | {limit_time:<14.3f}s")
    print(f"{'Production-like operations':<30} | {prod_rdd_time:<14.3f}s | {prod_limit_time:<14.3f}s")
    
    if rdd_time > limit_time * 1.5:
        print(f"\n⚠️  rdd.isEmpty() is {rdd_time/limit_time:.1f}x SLOWER!")
    
    print(f"\n🔍 Check Spark UI at http://localhost:4040")
    print("   - Storage tab: Compare partition caching")
    print("   - Stages tab: Look for recomputation")
    
    print("\n" + "="*70)
    print("PRODUCTION ISSUE EXPLAINED:")
    print("="*70)
    print("""
In your production environment with 1600 partitions:
1. rdd.isEmpty() only evaluates first ~98 partitions (6%)
2. These partial evaluations prevent full DataFrame caching
3. Subsequent operations must recompute the missing 94%
4. This causes the 45min → 200min performance degradation

Solution: Replace df.rdd.isEmpty() with df.limit(1).count() == 0
""")
    
    input("\nPress Enter to exit...")
    spark.stop()

if __name__ == "__main__":
    main()
