from __future__ import annotations

from pathlib import Path

from pyspark.sql import Row
from pyspark.sql.types import IntegerType, LongType, StructField, StructType

from common import ONLINE_TRAIN_PATH, OUTPUT_DIR, build_spark, write_df_as_single_csv


def main() -> None:
    spark = build_spark("task1_rdd_merchant_consumption", enable_hive=True)
    sc = spark.sparkContext

    rdd = sc.textFile(ONLINE_TRAIN_PATH)
    header = rdd.first()

    def parse(line: str):
        if line == header:
            return None
        parts = line.split(",")
        if len(parts) < 7:
            return None
        merchant_id = parts[1]
        coupon_id = parts[3]
        date = parts[6]

        try:
            mid = int(merchant_id)
        except ValueError:
            return None

        negative = 1 if (date == "null" and coupon_id != "null") else 0
        normal = 1 if (date != "null" and coupon_id == "null") else 0
        positive = 1 if (date != "null" and coupon_id != "null") else 0
        return mid, (negative, normal, positive)

    merged = (
        rdd.map(parse)
        .filter(lambda x: x is not None)
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
        .sortByKey(ascending=True)
    )

    schema = StructType(
        [
            StructField("Merchant_id", IntegerType(), False),
            StructField("Negative", LongType(), False),
            StructField("Normal", LongType(), False),
            StructField("Positive", LongType(), False),
        ]
    )

    df = spark.createDataFrame(
        merged.map(lambda kv: (kv[0], int(kv[1][0]), int(kv[1][1]), int(kv[1][2]))),
        schema=schema,
    )

    df.write.mode("overwrite").saveAsTable("online_consumption_table")

    head10 = df.orderBy("Merchant_id").limit(10).collect()
    for row in head10:
        print(f"{row['Merchant_id']} {row['Negative']} {row['Normal']} {row['Positive']}")

    out_csv = OUTPUT_DIR / "task1_online_consumption_table.csv"
    write_df_as_single_csv(df.orderBy("Merchant_id"), out_csv, header=True)

    out_head10 = OUTPUT_DIR / "task1_online_consumption_table_head10.txt"
    with out_head10.open("w", encoding="utf-8") as f:
        for row in head10:
            f.write(f"{row['Merchant_id']} {row['Negative']} {row['Normal']} {row['Positive']}\n")

    spark.stop()


if __name__ == "__main__":
    main()
