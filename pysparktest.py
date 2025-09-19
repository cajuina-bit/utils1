#!/usr/bin/env python3

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when
from pyspark.storagelevel import StorageLevel


def get_spark():
    return SparkSession.builder \
        .appName("DataFrame Empty Check Test") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()


def create_test_data(spark, partitions=50, rows_per_partition=5000):
    print(f"Setting up test data: {partitions} partitions, {rows_per_partition} rows each")

    df = spark.range(partitions * rows_per_partition) \
        .repartition(partitions) \
        .withColumn("value", rand()) \
        .withColumn("computed",
                   when(col("id") % 2 == 0, col("id") * col("value") * 1.7)
                   .otherwise(col("id") * col("value") * 2.1)) \
        .withColumn("group", (col("id") % 20).cast("string"))

    return df


def test_approach_1(df):
    print("\n--- Testing approach 1: rdd.isEmpty() ---")

    spark.catalog.clearCache()
    cached_df = df.persist(StorageLevel.MEMORY_ONLY)

    print("Persisting data...")
    start = time.time()
    count = cached_df.count()
    persist_time = time.time() - start
    print(f"Initial count: {count:,} rows ({persist_time:.2f}s)")

    print("Checking if empty using rdd.isEmpty()...")
    start = time.time()
    empty = cached_df.rdd.isEmpty()
    check_time = time.time() - start
    print(f"Empty check result: {empty} ({check_time:.3f}s)")

    operations = [
        ("Group by operation", lambda df: df.groupBy("group").count().count()),
        ("Aggregation", lambda df: df.agg({"computed": "sum"}).collect()[0][0]),
        ("Filter operation", lambda df: df.filter(col("computed") > 100).count()),
        ("Distinct groups", lambda df: df.select("group").distinct().count())
    ]

    times = []
    for name, op in operations:
        print(f"Running {name.lower()}...")
        start = time.time()
        result = op(cached_df)
        duration = time.time() - start
        times.append(duration)
        print(f"{name}: {duration:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"Average operation time: {avg_time:.3f}s")

    return {
        'check_time': check_time,
        'operation_times': times,
        'avg_time': avg_time,
        'persist_time': persist_time
    }


def test_approach_2(df):
    print("\n--- Testing approach 2: limit(1).count() ---")

    spark.catalog.clearCache()
    cached_df = df.persist(StorageLevel.MEMORY_ONLY)

    print("Persisting data...")
    start = time.time()
    count = cached_df.count()
    persist_time = time.time() - start
    print(f"Initial count: {count:,} rows ({persist_time:.2f}s)")

    print("Checking if empty using limit(1).count()...")
    start = time.time()
    empty = cached_df.limit(1).count() == 0
    check_time = time.time() - start
    print(f"Empty check result: {empty} ({check_time:.3f}s)")

    operations = [
        ("Group by operation", lambda df: df.groupBy("group").count().count()),
        ("Aggregation", lambda df: df.agg({"computed": "sum"}).collect()[0][0]),
        ("Filter operation", lambda df: df.filter(col("computed") > 100).count()),
        ("Distinct groups", lambda df: df.select("group").distinct().count())
    ]

    times = []
    for name, op in operations:
        print(f"Running {name.lower()}...")
        start = time.time()
        result = op(cached_df)
        duration = time.time() - start
        times.append(duration)
        print(f"{name}: {duration:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"Average operation time: {avg_time:.3f}s")

    return {
        'check_time': check_time,
        'operation_times': times,
        'avg_time': avg_time,
        'persist_time': persist_time
    }


def show_results(results1, results2):
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    print(f"Empty check timing:")
    print(f"  rdd.isEmpty():      {results1['check_time']:.3f}s")
    print(f"  limit(1).count():   {results2['check_time']:.3f}s")

    if results1['check_time'] > results2['check_time']:
        ratio = results1['check_time'] / results2['check_time']
        print(f"  limit(1).count() is {ratio:.1f}x faster for empty checks")

    print(f"\nSubsequent operations average:")
    print(f"  After rdd.isEmpty():      {results1['avg_time']:.3f}s")
    print(f"  After limit(1).count():   {results2['avg_time']:.3f}s")

    if results1['avg_time'] > results2['avg_time']:
        ratio = results1['avg_time'] / results2['avg_time']
        improvement = ((results1['avg_time'] - results2['avg_time']) / results1['avg_time']) * 100
        print(f"  limit(1).count() is {ratio:.1f}x faster ({improvement:.1f}% improvement)")

    print(f"\nOperation breakdown:")
    ops = ["Group by", "Aggregation", "Filter", "Distinct"]
    for i, op in enumerate(ops):
        t1, t2 = results1['operation_times'][i], results2['operation_times'][i]
        if t1 > t2:
            improvement = ((t1 - t2) / t1) * 100
            print(f"  {op}: {improvement:.1f}% faster with limit(1).count()")
        else:
            degradation = ((t2 - t1) / t2) * 100
            print(f"  {op}: {degradation:.1f}% slower with limit(1).count()")

    print(f"\nKey findings:")
    if results1['avg_time'] > results2['avg_time']:
        print(f"- rdd.isEmpty() appears to cause recomputation issues")
        print(f"- limit(1).count() maintains better cache utilization")
        print(f"- Performance difference suggests caching problems with rdd.isEmpty()")
    else:
        print(f"- Both approaches show similar performance")

    total_diff = sum(results1['operation_times']) - sum(results2['operation_times'])
    if total_diff > 0:
        print(f"- Total time saved with limit(1).count(): {total_diff:.2f}s")


if __name__ == "__main__":
    print("DataFrame Empty Check Performance Test")
    print("Testing impact on cached DataFrame operations")

    spark = get_spark()

    try:
        df = create_test_data(spark, partitions=50, rows_per_partition=5000)
        print(f"Test dataset: {df.rdd.getNumPartitions()} partitions")

        results1 = test_approach_1(df)
        time.sleep(1)
        results2 = test_approach_2(df)

        show_results(results1, results2)

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        spark.stop()
