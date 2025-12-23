from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from common import OUTPUT_DIR, build_spark, write_rdd_as_single_text


def main() -> None:
    spark = build_spark("task2_sql_merchant_positive_ratio", enable_hive=True)

    try:
        table = spark.table("online_consumption_table")
    except AnalysisException:
        table = spark.read.parquet(str(OUTPUT_DIR / "online_consumption_table.parquet"))

    ranked = (
        table.select(
            F.col("Merchant_id").alias("Merchant_id"),
            F.col("Positive").cast("double").alias("Positive"),
            (F.col("Negative") + F.col("Normal") + F.col("Positive")).cast("double").alias("Total"),
        )
        .where(F.col("Total") > 0)
        .select(
            "Merchant_id",
            (F.col("Positive") / F.col("Total")).alias("pos_ratio"),
            F.col("Positive").cast("long").alias("pos_cnt"),
            F.col("Total").cast("long").alias("total_cnt"),
        )
        .orderBy(F.col("pos_ratio").desc(), F.col("pos_cnt").desc(), F.col("Merchant_id").asc())
        .limit(10)
    )

    rows = ranked.collect()
    for r in rows:
        print(f"{r['Merchant_id']} {float(r['pos_ratio']):.6f} {r['pos_cnt']} {r['total_cnt']}")

    out_file = OUTPUT_DIR / "task2_top10_merchants_by_positive_ratio.txt"
    write_rdd_as_single_text(
        ranked.rdd.map(
            lambda r: f"{r['Merchant_id']} {float(r['pos_ratio']):.6f} {r['pos_cnt']} {r['total_cnt']}"
        ),
        out_file,
    )

    spark.stop()


if __name__ == "__main__":
    main()
