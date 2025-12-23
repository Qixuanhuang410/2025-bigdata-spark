from __future__ import annotations

from pyspark.sql import functions as F

from common import OFFLINE_TRAIN_PATH, OUTPUT_DIR, build_spark, read_csv_with_nulls, write_rdd_as_single_text


def main() -> None:
    spark = build_spark("task2_sql_coupon_time_distribution", enable_hive=False)

    df = read_csv_with_nulls(spark, OFFLINE_TRAIN_PATH)

    used = (
        df.select(
            F.col("Coupon_id").alias("Coupon_id"),
            F.to_date(F.col("Date"), "yyyyMMdd").alias("use_date"),
        )
        .where(F.col("Coupon_id").isNotNull() & F.col("use_date").isNotNull())
        .select("Coupon_id", F.dayofmonth("use_date").alias("d"))
    )

    dist = (
        used.groupBy("Coupon_id")
        .agg(
            (F.sum(F.when((F.col("d") >= 1) & (F.col("d") <= 10), 1).otherwise(0)) / F.count("*")).alias(
                "p_early"
            ),
            (F.sum(F.when((F.col("d") >= 11) & (F.col("d") <= 20), 1).otherwise(0)) / F.count("*")).alias(
                "p_mid"
            ),
            (F.sum(F.when(F.col("d") >= 21, 1).otherwise(0)) / F.count("*")).alias("p_late"),
            F.count("*").alias("total_used"),
        )
        .select(
            "Coupon_id",
            F.round("p_early", 6).alias("p_early"),
            F.round("p_mid", 6).alias("p_mid"),
            F.round("p_late", 6).alias("p_late"),
        )
        .orderBy("Coupon_id")
    )

    out_file = OUTPUT_DIR / "task2_coupon_time_distribution.txt"
    write_rdd_as_single_text(
        dist.rdd.map(lambda r: f"{r['Coupon_id']} {r['p_early']} {r['p_mid']} {r['p_late']}"),
        out_file,
    )

    print("Wrote", out_file)
    spark.stop()


if __name__ == "__main__":
    main()
