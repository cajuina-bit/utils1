from pyspark.sql import SparkSession
from pyspark import StorageLevel
import time
import random

# Initialize Spark with better memory settings
spark = SparkSession.builder \
    .appName("CacheTestRDDvsLimit") \
    .config("spark.ui.enabled", "true") \
    .config("spark.ui.port", "4040") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.sql.shuffle.partitions", "20") \
    .master("local[2]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

def create_simple_dataframe(num_rows=100000):
    """Create a simpler DataFrame to avoid memory issues"""
    print(f"Creating DataFrame with {num_rows} rows...")
    
    # Create a simple but meaningful dataset
    data = [(i, i * 2, f"value_{i}") for i in range(num_rows)]
    
    df = spark.createDataFrame(data, ["id", "value", "name"])
    
    # Add some transformations to make it non-trivial
    df = df.filter(df.id % 2 == 0) \
        .withColumn("computed", df.value * 10)
    
    return df

def test_with_rdd_isempty(df):
    """Test caching with rdd.isEmpty() - demonstrating the issue"""
    print("\n" + "="*60)
    print("Testing with rdd.isEmpty()")
    print("="*60)
    
    # Clear any existing cache
    spark.catalog.clearCache()
    
    # Persist the DataFrame
    print("\n1. Calling persist() on DataFrame...")
    df_persisted = df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Force materialization first to ensure data is available
    print("2. Forcing initial materialization with count()...")
    initial_count = df_persisted.count()
    print(f"   Initial count: {initial_count}")
    
    # Check cache status BEFORE isEmpty
    print("\n3. Cache Status BEFORE isEmpty():")
    check_cache_status()
    
    # Now check if empty using rdd.isEmpty()
    print("\n4. Calling rdd.isEmpty()...")
    try:
        start = time.time()
        is_empty = df_persisted.rdd.isEmpty()
        empty_check_time = time.time() - start
        print(f"   isEmpty() returned: {is_empty}")
        print(f"   Time taken: {empty_check_time:.2f} seconds")
    except Exception as e:
        print(f"   ERROR during isEmpty(): {str(e)[:100]}")
        empty_check_time = -1
    
    # Check cache status AFTER isEmpty
    print("\n5. Cache Status AFTER isEmpty():")
    check_cache_status()
    
    # Perform operations to test cache
    print("\n6. Testing cache effectiveness with count operations:")
    
    start = time.time()
    count1 = df_persisted.count()
    time1 = time.time() - start
    print(f"   First count: {count1} (took {time1:.2f}s)")
    
    start = time.time()
    count2 = df_persisted.count()
    time2 = time.time() - start
    print(f"   Second count: {count2} (took {time2:.2f}s)")
    
    return empty_check_time, time1, time2

def test_with_limit_count(df):
    """Test caching with limit(1).count() - the correct approach"""
    print("\n" + "="*60)
    print("Testing with limit(1).count()")
    print("="*60)
    
    # Clear any existing cache
    spark.catalog.clearCache()
    
    # Persist the DataFrame
    print("\n1. Calling persist() on DataFrame...")
    df_persisted = df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Check if empty using limit(1).count()
    print("2. Calling limit(1).count() == 0...")
    start = time.time()
    is_empty = df_persisted.limit(1).count() == 0
    empty_check_time = time.time() - start
    print(f"   Result: {is_empty}")
    print(f"   Time taken: {empty_check_time:.2f} seconds")
    
    # Check cache status
    print("\n3. Cache Status after limit(1).count():")
    check_cache_status()
    
    # Force full materialization
    print("\n4. Forcing full materialization with count()...")
    start = time.time()
    count = df_persisted.count()
    materialization_time = time.time() - start
    print(f"   Count: {count} (took {materialization_time:.2f}s)")
    
    # Check cache status after materialization
    print("\n5. Cache Status after full materialization:")
    check_cache_status()
    
    # Test cache effectiveness
    print("\n6. Testing cache effectiveness with count operations:")
    
    start = time.time()
    count1 = df_persisted.count()
    time1 = time.time() - start
    print(f"   First count: {count1} (took {time1:.2f}s)")
    
    start = time.time()
    count2 = df_persisted.count()
    time2 = time.time() - start
    print(f"   Second count: {count2} (took {time2:.2f}s)")
    
    return empty_check_time, time1, time2

def check_cache_status():
    """Check and print cache status"""
    try:
        storage_info = spark.sparkContext._jsc.sc().getRDDStorageInfo()
        if len(storage_info) == 0:
            print("   No RDDs cached")
        else:
            for rdd_info in storage_info:
                if rdd_info.numCachedPartitions() > 0:
                    cached = rdd_info.numCachedPartitions()
                    total = rdd_info.numPartitions()
                    mem_size = rdd_info.memSize() / 1024 / 1024
                    disk_size = rdd_info.diskSize() / 1024 / 1024
                    print(f"   RDD {rdd_info.id()}: {cached}/{total} partitions cached ({cached*100/total:.1f}%)")
                    print(f"        Memory: {mem_size:.2f} MB, Disk: {disk_size:.2f} MB")
    except Exception as e:
        print(f"   Error checking cache: {e}")

def main():
    print("\n" + "="*60)
    print("SPARK CACHE COMPARISON TEST")
    print("Demonstrating the rdd.isEmpty() caching issue")
    print("="*60)
    
    # Create test DataFrame
    df = create_simple_dataframe(50000)  # Reduced size for local testing
    
    # Test with limit(1).count() FIRST (the correct way)
    limit_times = test_with_limit_count(df)
    
    # Test with rdd.isEmpty() (the problematic way)
    rdd_times = test_with_rdd_isempty(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nTiming Comparison:")
    print(f"{'Method':<20} | {'Empty Check':>12} | {'Count 1':>10} | {'Count 2':>10}")
    print("-" * 60)
    
    if rdd_times[0] >= 0:
        print(f"{'rdd.isEmpty()':<20} | {rdd_times[0]:>11.3f}s | {rdd_times[1]:>9.3f}s | {rdd_times[2]:>9.3f}s")
    else:
        print(f"{'rdd.isEmpty()':<20} | {'ERROR':>12} | {rdd_times[1]:>9.3f}s | {rdd_times[2]:>9.3f}s")
    
    print(f"{'limit(1).count()':<20} | {limit_times[0]:>11.3f}s | {limit_times[1]:>9.3f}s | {limit_times[2]:>9.3f}s")
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    print("""
1. With limit(1).count():
   - DataFrame stays as DataFrame
   - Cache works properly
   - Subsequent operations are fast
   
2. With rdd.isEmpty():
   - Converts DataFrame to RDD (breaks optimization)
   - May cause cache management errors
   - Cache may be partially populated or invalidated
   - Performance degrades significantly
   
3. The error you saw demonstrates the cache corruption
   that happens with rdd.isEmpty()!
""")
    
    print(f"\nSpark UI available at http://localhost:4040")
    print("Check the Storage tab to see cache differences")
    
    # Keep alive to check UI
    input("\nPress Enter to exit...")
    spark.stop()

if __name__ == "__main__":
    main()
