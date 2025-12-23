from __future__ import annotations

from pathlib import Path

from common import ONLINE_TRAIN_PATH, OUTPUT_DIR, build_spark, write_rdd_as_single_text


def main() -> None:
    spark = build_spark("task1_rdd_coupon_usage", enable_hive=False)
    sc = spark.sparkContext

    rdd = sc.textFile(ONLINE_TRAIN_PATH)
    header = rdd.first()

    def parse(line: str):
        if line == header:
            return None
        parts = line.split(",")
        if len(parts) < 7:
            return None
        coupon_id = parts[3]
        date = parts[6]
        if coupon_id == "null" or date == "null":
            return None
        return coupon_id

    counts = (
        rdd.map(parse)
        .filter(lambda x: x is not None)
        .map(lambda cid: (cid, 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda kv: kv[1], ascending=False)
    )

    top10 = counts.take(10)
    for cid, c in top10:
        print(f"{cid} {c}")

    out_file = OUTPUT_DIR / "task1_coupon_usage_counts.txt"
    write_rdd_as_single_text(counts.map(lambda kv: f"{kv[0]} {kv[1]}"), out_file)

    out_top10 = OUTPUT_DIR / "task1_coupon_usage_counts_top10.txt"
    with out_top10.open("w", encoding="utf-8") as f:
        for cid, c in top10:
            f.write(f"{cid} {c}\n")

    spark.stop()


if __name__ == "__main__":
    main()
