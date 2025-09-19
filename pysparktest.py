from pyspark.sql import SparkSession
import time
import random

# Initialize Spark
spark = SparkSession.builder \
    .appName("CacheTestRDDvsLimit") \
    .config("spark.ui.enabled", "true") \
    .config("spark.ui.port", "4040") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .master("local[4]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

def create_expensive_dataframe(num_rows=1000000):
    """Create a DataFrame that's expensive to compute"""
    print(f"Creating DataFrame with {num_rows} rows...")
    
    # Create an expensive computation - simulate complex transformations
    def expensive_map(x):
        # Simulate expensive computation
        time.sleep(0.0001)  # Small delay to simulate work
        return (x, x * 2, x * 3, str(x), random.random())
    
    rdd = spark.sparkContext.parallelize(range(num_rows), 100) \
        .map(expensive_map)
    
    columns = ["id", "value1", "value2", "str_value", "random_value"]
    df = spark.createDataFrame(rdd, columns)
    
    # Add more transformations to make it expensive
    df = df.filter(df.id % 2 == 0) \
        .withColumn("computed", df.value1 + df.value2) \
        .withColumn("concat", df.str_value)
    
    return df

def test_with_rdd_isempty(df):
    """Test caching with rdd.isEmpty()"""
    print("\n=== Testing with rdd.isEmpty() ===")
    
    # Clear any existing cache
    spark.catalog.clearCache()
    
    # Persist the DataFrame
    df_persisted = df.persist()
    print(f"Called persist() on DataFrame")
    
    # Check if empty using rdd.isEmpty()
    start = time.time()
    is_empty = df_persisted.rdd.isEmpty()
    empty_check_time = time.time() - start
    print(f"isEmpty() check took: {empty_check_time:.2f} seconds")
    print(f"DataFrame is empty: {is_empty}")
    
    # Check cache status
    print("\nCache Status after isEmpty():")
    for rdd_info in spark.sparkContext._jsc.sc().getRDDStorageInfo():
        if rdd_info.numCachedPartitions() > 0:
            print(f"  RDD {rdd_info.id()}: {rdd_info.numCachedPartitions()}/{rdd_info.numPartitions()} partitions cached")
            print(f"  Memory Size: {rdd_info.memSize() / 1024 / 1024:.2f} MB")
            print(f"  Disk Size: {rdd_info.diskSize() / 1024 / 1024:.2f} MB")
    
    # Now perform actual computation
    print("\nPerforming count operation...")
    start = time.time()
    count = df_persisted.count()
    count_time = time.time() - start
    print(f"Count took: {count_time:.2f} seconds")
    print(f"Row count: {count}")
    
    # Perform another operation to see if cache is used
    print("\nPerforming second count operation (should use cache)...")
    start = time.time()
    count2 = df_persisted.count()
    count2_time = time.time() - start
    print(f"Second count took: {count2_time:.2f} seconds")
    
    return empty_check_time, count_time, count2_time

def test_with_limit_count(df):
    """Test caching with limit(1).count()"""
    print("\n=== Testing with limit(1).count() ===")
    
    # Clear any existing cache
    spark.catalog.clearCache()
    
    # Persist the DataFrame
    df_persisted = df.persist()
    print(f"Called persist() on DataFrame")
    
    # Check if empty using limit(1).count()
    start = time.time()
    is_empty = df_persisted.limit(1).count() == 0
    empty_check_time = time.time() - start
    print(f"limit(1).count() check took: {empty_check_time:.2f} seconds")
    print(f"DataFrame is empty: {is_empty}")
    
    # Check cache status
    print("\nCache Status after limit(1).count():")
    for rdd_info in spark.sparkContext._jsc.sc().getRDDStorageInfo():
        if rdd_info.numCachedPartitions() > 0:
            print(f"  RDD {rdd_info.id()}: {rdd_info.numCachedPartitions()}/{rdd_info.numPartitions()} partitions cached")
            print(f"  Memory Size: {rdd_info.memSize() / 1024 / 1024:.2f} MB")
            print(f"  Disk Size: {rdd_info.diskSize() / 1024 / 1024:.2f} MB")
    
    # Now perform actual computation
    print("\nPerforming count operation...")
    start = time.time()
    count = df_persisted.count()
    count_time = time.time() - start
    print(f"Count took: {count_time:.2f} seconds")
    print(f"Row count: {count}")
    
    # Perform another operation to see if cache is used
    print("\nPerforming second count operation (should use cache)...")
    start = time.time()
    count2 = df_persisted.count()
    count2_time = time.time() - start
    print(f"Second count took: {count2_time:.2f} seconds")
    
    return empty_check_time, count_time, count2_time

def main():
    print("Starting Spark Cache Comparison Test")
    print("=" * 50)
    
    # Create expensive DataFrame
    df = create_expensive_dataframe(500000)  # Adjust size based on your machine
    
    # Test with rdd.isEmpty()
    rdd_times = test_with_rdd_isempty(df)
    
    print("\n" + "=" * 50)
    
    # Test with limit(1).count()
    limit_times = test_with_limit_count(df)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Method              | Empty Check | First Count | Second Count")
    print(f"rdd.isEmpty()       | {rdd_times[0]:11.2f}s | {rdd_times[1]:11.2f}s | {rdd_times[2]:12.2f}s")
    print(f"limit(1).count()    | {limit_times[0]:11.2f}s | {limit_times[1]:11.2f}s | {limit_times[2]:12.2f}s")
    
    print("\nKEY OBSERVATIONS:")
    print("- With rdd.isEmpty(): Check cache status - likely shows partial caching")
    print("- With limit(1).count(): Should show full caching after first count")
    print("- Second count with limit(1).count() should be much faster (reading from cache)")
    print("- Second count with rdd.isEmpty() might be slower (recomputation)")
    
    print(f"\nSpark UI available at http://localhost:4040")
    print("Check the Storage tab to see cache utilization")
    
    # Keep alive to check UI
    input("\nPress Enter to exit and stop Spark...")
    spark.stop()

if __name__ == "__main__":
    main()
