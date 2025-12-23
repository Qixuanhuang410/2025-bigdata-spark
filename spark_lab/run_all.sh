#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/huang"

SPARK_SUBMIT="${SPARK_SUBMIT:-}"
if [[ -z "$SPARK_SUBMIT" ]]; then
	if [[ -n "${SPARK_HOME:-}" ]]; then
		SPARK_SUBMIT="$SPARK_HOME/bin/spark-submit"
	else
		SPARK_SUBMIT="/home/huang/tools/spark/bin/spark-submit"
	fi
fi

if [[ ! -x "$SPARK_SUBMIT" ]]; then
	echo "ERROR: spark-submit not found or not executable: $SPARK_SUBMIT" >&2
	echo "Set SPARK_HOME or SPARK_SUBMIT, e.g.: export SPARK_HOME=/path/to/spark" >&2
	exit 1
fi

export PYSPARK_PYTHON=/usr/bin/python3

cd "$ROOT"

$SPARK_SUBMIT spark_lab/task1_rdd_coupon_usage.py
$SPARK_SUBMIT spark_lab/task1_rdd_merchant_consumption.py
$SPARK_SUBMIT spark_lab/task2_sql_coupon_time_distribution.py
$SPARK_SUBMIT spark_lab/task2_sql_merchant_positive_ratio.py
$SPARK_SUBMIT spark_lab/task3_mllib_predict.py
