from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

from pyspark.sql import SparkSession, DataFrame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPARK_WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"
SPARK_TMP_DIR = PROJECT_ROOT / "spark-tmp"
SPARK_METASTORE_DB_DIR = PROJECT_ROOT / "spark-metastore" / "metastore_db"

def _as_file_uri(path: Path) -> str:
    return f"file://{path.resolve()}"


ONLINE_TRAIN_PATH = _as_file_uri(DATA_DIR / "ccf_online_stage1_train.csv.gz")
OFFLINE_TRAIN_PATH = _as_file_uri(DATA_DIR / "ccf_offline_stage1_train.csv.gz")
OFFLINE_TEST_PATH = _as_file_uri(DATA_DIR / "ccf_offline_stage1_test_revised.csv")


def build_spark(app_name: str, enable_hive: bool = True) -> SparkSession:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPARK_WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    SPARK_TMP_DIR.mkdir(parents=True, exist_ok=True)
    SPARK_METASTORE_DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.warehouse.dir", str(SPARK_WAREHOUSE_DIR))
        .config("spark.local.dir", str(SPARK_TMP_DIR))
        .config("spark.sql.session.timeZone", "Asia/Shanghai")
        .config("spark.hadoop.fs.defaultFS", "file:///")
    )

    if enable_hive:
        # Use a dedicated embedded Derby metastore to avoid locking/conflicts with any
        # existing metastore_db directories in the working directory.
        builder = builder.config(
            "spark.hadoop.javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={SPARK_METASTORE_DB_DIR};create=true",
        ).config("spark.hadoop.datanucleus.schema.autoCreateAll", "true")
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_csv_with_nulls(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("nullValue", "null")
        .csv(path)
    )


def _take_single_part_file(dir_path: Path) -> Path:
    part_files = sorted(dir_path.glob("part-*") )
    if not part_files:
        raise FileNotFoundError(f"No part-* files found in {dir_path}")
    return part_files[0]


def write_single_text(lines: Iterable[str], out_file: Path) -> None:
    """Write an iterable of lines as a single local file."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line))
            if not str(line).endswith("\n"):
                f.write("\n")


def write_rdd_as_single_text(rdd, out_file: Path, tmp_dir: Optional[Path] = None) -> None:
    """Save an RDD[String] as a single local file."""
    out_file = out_file.resolve()
    tmp_dir = (tmp_dir or out_file.with_suffix(out_file.suffix + ".dir")).resolve()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    rdd.coalesce(1).saveAsTextFile(str(tmp_dir))

    part_file = _take_single_part_file(tmp_dir)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(part_file, out_file)
    shutil.rmtree(tmp_dir)


def write_df_as_single_csv(df: DataFrame, out_file: Path, header: bool = True) -> None:
    """Write a DataFrame to a single CSV file (local FS) via a temporary directory."""
    out_file = out_file.resolve()
    tmp_dir = out_file.with_suffix(out_file.suffix + ".dir").resolve()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", header)
        .csv(str(tmp_dir))
    )

    part_file = _take_single_part_file(tmp_dir)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(part_file, out_file)
    shutil.rmtree(tmp_dir)
