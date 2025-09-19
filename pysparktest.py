#!/usr/bin/env python3

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
from pyspark.storagelevel import StorageLevel


def get_spark():
    return SparkSession.builder \
        .appName("Storage UI Cache Test") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .config("spark.storage.memoryFraction", "0.8") \
        .getOrCreate()


def create_data(spark):
    partitions = 50
    rows_per_partition = 10000

    print(f"Creating {partitions} partitions with {rows_per_partition} rows each")

    df = spark.range(partitions * rows_per_partition) \
        .repartition(partitions) \
        .withColumn("value", rand()) \
        .withColumn("computed", col("id") * col("value") * 100)

    return df


def test_rdd_isEmpty_storage(spark, df):
    print("\n=== Testing rdd.isEmpty() - Check Storage tab ===")

    spark.catalog.clearCache()

    cached_df = df.persist(StorageLevel.MEMORY_ONLY)
    cached_df.cache()

    print("1. DataFrame persisted and cached")
    print("2. Triggering initial cache population...")

    count = cached_df.count()
    print(f"   Count: {count:,} rows")

    print("3. Check Spark UI Storage tab now - should show full caching")
    input("   Press Enter when you've checked the Storage tab...")

    print("4. Now calling rdd.isEmpty()...")
    is_empty = cached_df.rdd.isEmpty()
    print(f"   Result: {is_empty}")

    print("5. Check Spark UI Storage tab again - look for partial caching!")
    print("   You should see fewer cached partitions now")
    input("   Press Enter after checking Storage tab...")

    print("6. Running another operation to see recomputation...")
    start = time.time()
    group_count = cached_df.groupBy("computed").count().count()
    duration = time.time() - start
    print(f"   GroupBy operation: {duration:.3f}s")

    print("7. Final check - Storage tab should show the impact")
    input("   Press Enter after final Storage tab check...")


def test_limit_count_storage(spark, df):
    print("\n=== Testing limit(1).count() - Check Storage tab ===")

    spark.catalog.clearCache()

    cached_df = df.persist(StorageLevel.MEMORY_ONLY)
    cached_df.cache()

    print("1. DataFrame persisted and cached")
    print("2. Triggering initial cache population...")

    count = cached_df.count()
    print(f"   Count: {count:,} rows")

    print("3. Check Spark UI Storage tab now - should show full caching")
    input("   Press Enter when you've checked the Storage tab...")

    print("4. Now calling limit(1).count()...")
    is_empty = cached_df.limit(1).count() == 0
    print(f"   Result: {is_empty}")

    print("5. Check Spark UI Storage tab again - should maintain full caching")
    input("   Press Enter after checking Storage tab...")

    print("6. Running another operation...")
    start = time.time()
    group_count = cached_df.groupBy("computed").count().count()
    duration = time.time() - start
    print(f"   GroupBy operation: {duration:.3f}s")

    print("7. Final check - Storage tab should show maintained caching")
    input("   Press Enter after final Storage tab check...")


def main():
    print("Storage UI Cache Visualization Test")
    print("This will help you see partial caching on Spark UI Storage tab")
    print("\nOpen Spark UI at: http://localhost:4040")
    print("Navigate to Storage tab to see cache details")

    spark = get_spark()

    try:
        df = create_data(spark)

        choice = input("\nWhich test? (1=rdd.isEmpty, 2=limit.count, 3=both): ")

        if choice in ['1', '3']:
            test_rdd_isEmpty_storage(spark, df)

        if choice in ['2', '3']:
            test_limit_count_storage(spark, df)

        print("\nTest complete!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Keeping Spark session alive for UI inspection...")
        input("Press Enter to shutdown Spark...")
        spark.stop()


if __name__ == "__main__":
    main()
