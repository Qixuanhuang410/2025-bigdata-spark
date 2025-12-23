from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import FeatureHasher, VectorAssembler
from pyspark.sql import functions as F

from common import (
    OFFLINE_TEST_PATH,
    OFFLINE_TRAIN_PATH,
    OUTPUT_DIR,
    build_spark,
    read_csv_with_nulls,
    write_df_as_single_csv,
)


def add_features(df):
    # Parse dates
    df = df.withColumn("recv_date", F.to_date(F.col("Date_received"), "yyyyMMdd"))
    if "Date" in df.columns:
        df = df.withColumn("use_date", F.to_date(F.col("Date"), "yyyyMMdd"))

    # Distance
    df = df.withColumn(
        "distance_f",
        F.when(F.col("Distance").isNull(), F.lit(-1.0)).otherwise(F.col("Distance").cast("double")),
    )

    # Discount parsing
    is_manjian = F.instr(F.col("Discount_rate"), ":") > 0
    threshold = F.split(F.col("Discount_rate"), ":").getItem(0).cast("double")
    reduction = F.split(F.col("Discount_rate"), ":").getItem(1).cast("double")

    discount_rate_f = F.when(
        is_manjian & threshold.isNotNull() & (threshold > 0) & reduction.isNotNull(),
        (threshold - reduction) / threshold,
    ).otherwise(F.col("Discount_rate").cast("double"))

    df = df.withColumn("is_manjian", F.when(is_manjian, F.lit(1.0)).otherwise(F.lit(0.0)))
    df = df.withColumn("manjian_threshold", F.when(is_manjian, threshold).otherwise(F.lit(0.0)))
    df = df.withColumn("manjian_reduction", F.when(is_manjian, reduction).otherwise(F.lit(0.0)))
    df = df.withColumn("discount_rate_f", F.when(discount_rate_f.isNull(), F.lit(1.0)).otherwise(discount_rate_f))

    # Receive time features
    df = df.withColumn("recv_day", F.dayofmonth("recv_date").cast("double"))
    df = df.withColumn("recv_dow", F.dayofweek("recv_date").cast("double"))

    return df


def main() -> None:
    spark = build_spark("task3_mllib_predict", enable_hive=False)

    train_raw = read_csv_with_nulls(spark, OFFLINE_TRAIN_PATH)

    # Only coupon-received samples; do not leak July test period into training.
    train = (
        train_raw.where(F.col("Coupon_id").isNotNull() & F.col("Date_received").isNotNull())
        .withColumn("recv_date", F.to_date(F.col("Date_received"), "yyyyMMdd"))
        .where(F.col("recv_date") < F.lit("2016-07-01"))
        .drop("recv_date")
    )

    train = add_features(train)

    # Label: used within 15 days after receiving.
    train = train.withColumn(
        "label",
        F.when(
            F.col("use_date").isNotNull() & (F.datediff(F.col("use_date"), F.col("recv_date")) <= 15) & (F.datediff(F.col("use_date"), F.col("recv_date")) >= 0),
            F.lit(1.0),
        ).otherwise(F.lit(0.0)),
    )

    feature_hasher = FeatureHasher(
        inputCols=["User_id", "Merchant_id", "Coupon_id"],
        outputCol="hashed",
        numFeatures=1 << 18,
    )

    assembler = VectorAssembler(
        inputCols=[
            "hashed",
            "discount_rate_f",
            "is_manjian",
            "manjian_threshold",
            "manjian_reduction",
            "distance_f",
            "recv_day",
            "recv_dow",
        ],
        outputCol="features",
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.01,
    )

    pipeline = Pipeline(stages=[feature_hasher, assembler, lr])

    train_df, valid_df = train.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)

    valid_pred = model.transform(valid_df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(valid_pred)
    print(f"Validation AUC: {auc:.6f}")

    # Fit on full train and predict July test.
    full_model = pipeline.fit(train)

    test_raw = read_csv_with_nulls(spark, OFFLINE_TEST_PATH)
    test = test_raw.where(F.col("Coupon_id").isNotNull() & F.col("Date_received").isNotNull())
    test = add_features(test)

    pred = full_model.transform(test)

    out = (
        pred.select(
            F.col("User_id"),
            F.col("Coupon_id"),
            F.col("Date_received"),
            F.col("probability").getItem(1).alias("Probability"),
        )
        .withColumn("Probability", F.round(F.col("Probability"), 6))
    )

    out_file = OUTPUT_DIR / "task3_predictions.csv"
    write_df_as_single_csv(out, out_file, header=True)

    print("Wrote", out_file)
    spark.stop()


if __name__ == "__main__":
    main()
