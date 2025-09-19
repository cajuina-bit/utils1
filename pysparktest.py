#!/usr/bin/env python3

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when
from pyspark.storagelevel import StorageLevel


def get_spark():
    return SparkSession.builder \
        .appName("EmptyRDD Partial Caching Test") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .getOrCreate()


def create_large_dataset(spark, partitions=100, rows_per_partition=5000):
    print(f"Creating dataset: {partitions} partitions, {rows_per_partition} rows each")

    df = spark.range(partitions * rows_per_partition) \
        .repartition(partitions) \
        .withColumn("value", rand()) \
        .withColumn("computed", col("id") * col("value") * 2.5) \
        .withColumn("category", (col("id") % 25).cast("string"))

    return df


def get_cache_info(spark):
    try:
        sc = spark.sparkContext
        storage_status = sc._jsc.sc().getRDDStorageInfo()

        total_partitions = 0
        cached_partitions = 0
        memory_used = 0

        for rdd_info in storage_status:
            if hasattr(rdd_info, 'numPartitions'):
                total_partitions += rdd_info.numPartitions()
            if hasattr(rdd_info, 'numCachedPartitions'):
                cached_partitions += rdd_info.numCachedPartitions()
            if hasattr(rdd_info, 'memoryUsed'):
                memory_used += rdd_info.memoryUsed()

        cache_percentage = (cached_partitions / total_partitions * 100) if total_partitions > 0 else 0

        return {
            'total_partitions': total_partitions,
            'cached_partitions': cached_partitions,
            'cache_percentage': cache_percentage,
            'memory_mb': memory_used / (1024 * 1024)
        }
    except:
        return {'total_partitions': 0, 'cached_partitions': 0, 'cache_percentage': 0, 'memory_mb': 0}


def test_empty_rdd_caching(spark, df):
    print("\n" + "="*70)
    print("TESTING: emptyRDD() impact on caching")
    print("="*70)

    spark.catalog.clearCache()

    cached_df = df.persist(StorageLevel.MEMORY_ONLY)
    print("DataFrame persisted...")

    print("Step 1: Initial count to populate cache")
    start = time.time()
    count = cached_df.count()
    count_time = time.time() - start
    print(f"Count: {count:,} rows ({count_time:.3f}s)")

    time.sleep(0.5)
    cache_info = get_cache_info(spark)
    print(f"Cache after count: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    print("\nStep 2: Creating and checking emptyRDD")
    empty_rdd = spark.sparkContext.emptyRDD()
    print(f"EmptyRDD created with {empty_rdd.getNumPartitions()} partitions")

    start = time.time()
    is_empty = empty_rdd.isEmpty()
    empty_check_time = time.time() - start
    print(f"EmptyRDD.isEmpty(): {is_empty} ({empty_check_time:.3f}s)")

    time.sleep(0.5)
    cache_info = get_cache_info(spark)
    print(f"Cache after emptyRDD check: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    print("\nStep 3: Checking main DataFrame isEmpty via RDD")
    start = time.time()
    df_empty = cached_df.rdd.isEmpty()
    df_empty_time = time.time() - start
    print(f"DataFrame.rdd.isEmpty(): {df_empty} ({df_empty_time:.3f}s)")

    time.sleep(0.5)
    cache_info = get_cache_info(spark)
    print(f"Cache after DataFrame isEmpty: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    print("\nStep 4: Subsequent operations performance")
    operations = [
        ("Simple count", lambda df: df.count()),
        ("Group by", lambda df: df.groupBy("category").count().count()),
        ("Filter count", lambda df: df.filter(col("computed") > 100).count()),
        ("Aggregation", lambda df: df.agg({"computed": "sum"}).collect()[0][0])
    ]

    operation_times = []
    for name, op in operations:
        start = time.time()
        result = op(cached_df)
        duration = time.time() - start
        operation_times.append(duration)
        print(f"{name}: {duration:.3f}s")

        cache_info = get_cache_info(spark)
        print(f"  Cache: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    return {
        'count_time': count_time,
        'empty_check_time': empty_check_time,
        'df_empty_time': df_empty_time,
        'operation_times': operation_times,
        'avg_operation_time': sum(operation_times) / len(operation_times)
    }


def test_normal_caching(spark, df):
    print("\n" + "="*70)
    print("TESTING: Normal caching without emptyRDD interference")
    print("="*70)

    spark.catalog.clearCache()

    cached_df = df.persist(StorageLevel.MEMORY_ONLY)
    print("DataFrame persisted...")

    print("Step 1: Initial count to populate cache")
    start = time.time()
    count = cached_df.count()
    count_time = time.time() - start
    print(f"Count: {count:,} rows ({count_time:.3f}s)")

    time.sleep(0.5)
    cache_info = get_cache_info(spark)
    print(f"Cache after count: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    print("\nStep 2: Using limit(1).count() for empty check")
    start = time.time()
    is_empty = cached_df.limit(1).count() == 0
    limit_check_time = time.time() - start
    print(f"DataFrame.limit(1).count() == 0: {is_empty} ({limit_check_time:.3f}s)")

    time.sleep(0.5)
    cache_info = get_cache_info(spark)
    print(f"Cache after limit check: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    print("\nStep 3: Subsequent operations performance")
    operations = [
        ("Simple count", lambda df: df.count()),
        ("Group by", lambda df: df.groupBy("category").count().count()),
        ("Filter count", lambda df: df.filter(col("computed") > 100).count()),
        ("Aggregation", lambda df: df.agg({"computed": "sum"}).collect()[0][0])
    ]

    operation_times = []
    for name, op in operations:
        start = time.time()
        result = op(cached_df)
        duration = time.time() - start
        operation_times.append(duration)
        print(f"{name}: {duration:.3f}s")

        cache_info = get_cache_info(spark)
        print(f"  Cache: {cache_info['cached_partitions']}/{cache_info['total_partitions']} partitions ({cache_info['cache_percentage']:.1f}%)")

    return {
        'count_time': count_time,
        'limit_check_time': limit_check_time,
        'operation_times': operation_times,
        'avg_operation_time': sum(operation_times) / len(operation_times)
    }


def compare_results(empty_rdd_results, normal_results):
    print("\n" + "="*70)
    print("COMPARISON: EmptyRDD Impact vs Normal Caching")
    print("="*70)

    print("Empty check performance:")
    print(f"  With emptyRDD interference: {empty_rdd_results['df_empty_time']:.3f}s")
    print(f"  Normal limit(1).count():     {normal_results['limit_check_time']:.3f}s")

    if empty_rdd_results['df_empty_time'] > normal_results['limit_check_time']:
        ratio = empty_rdd_results['df_empty_time'] / normal_results['limit_check_time']
        print(f"  EmptyRDD approach is {ratio:.1f}x slower for empty checking")

    print(f"\nSubsequent operations average:")
    print(f"  After emptyRDD interference: {empty_rdd_results['avg_operation_time']:.3f}s")
    print(f"  Normal caching:               {normal_results['avg_operation_time']:.3f}s")

    if empty_rdd_results['avg_operation_time'] > normal_results['avg_operation_time']:
        ratio = empty_rdd_results['avg_operation_time'] / normal_results['avg_operation_time']
        improvement = ((empty_rdd_results['avg_operation_time'] - normal_results['avg_operation_time']) / empty_rdd_results['avg_operation_time']) * 100
        print(f"  Normal caching is {ratio:.1f}x faster ({improvement:.1f}% improvement)")

    print(f"\nOperation breakdown:")
    operations = ["Simple count", "Group by", "Filter count", "Aggregation"]
    for i, op in enumerate(operations):
        t1 = empty_rdd_results['operation_times'][i]
        t2 = normal_results['operation_times'][i]
        if t1 > t2:
            improvement = ((t1 - t2) / t1) * 100
            print(f"  {op}: {improvement:.1f}% faster with normal caching")
        else:
            degradation = ((t2 - t1) / t2) * 100
            print(f"  {op}: {degradation:.1f}% slower with normal caching")

    total_time_diff = sum(empty_rdd_results['operation_times']) - sum(normal_results['operation_times'])
    if total_time_diff > 0:
        print(f"\nTotal time saved with normal caching: {total_time_diff:.2f}s")

    print(f"\nKey insights:")
    print(f"- EmptyRDD operations may interfere with DataFrame caching")
    print(f"- Using rdd.isEmpty() can cause partial cache invalidation")
    print(f"- Normal DataFrame operations maintain better cache consistency")


def main():
    print("EmptyRDD Partial Caching Test")
    print("Investigating how emptyRDD() affects DataFrame caching")

    spark = get_spark()

    try:
        print(f"\nSpark UI: http://localhost:4040")
        print("Check Storage tab for cache details")

        df = create_large_dataset(spark, partitions=100, rows_per_partition=5000)
        print(f"Test dataset: {df.rdd.getNumPartitions()} partitions")

        empty_rdd_results = test_empty_rdd_caching(spark, df)
        time.sleep(2)
        normal_results = test_normal_caching(spark, df)

        compare_results(empty_rdd_results, normal_results)

        print(f"\nTest complete. Check Spark UI for detailed cache information.")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("Press Enter to shutdown Spark (check UI first)...")
        spark.stop()


if __name__ == "__main__":
    main()
