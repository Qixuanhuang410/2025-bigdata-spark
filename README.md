# 实验4:Spark 编程实验

本仓库包含 Spark RDD / Spark SQL / MLlib 三部分实验代码与运行结果（输出文件在 `outputs/`）。

- 实验报告：见 [实验报告.md](实验报告.md)
- 代码：见 `spark_lab/`
- 结果附件：见 `outputs/`
- 结果截图：见 `screenshots/`

## 目录结构

- `spark_lab/`：所有任务脚本与公共工具
- `outputs/`：任务输出文件
- `screenshots/`:实验结果截图

## 运行环境（本次实验实际环境）

- OS：Linux (Ubuntu 22.04)
- Java：OpenJDK 1.8
- Spark：3.5.1（binary 包，本地 `local[*]` 模式）

> 数据集 CSV 的空值以字符串 `null` 表示。

## 如何运行（示例）

假设你的 Spark 可执行为：`$SPARK_HOME/bin/spark-submit`。

```bash
$SPARK_HOME/bin/spark-submit spark_lab/task1_rdd_coupon_usage.py
$SPARK_HOME/bin/spark-submit spark_lab/task1_rdd_merchant_consumption.py
$SPARK_HOME/bin/spark-submit spark_lab/task2_sql_coupon_time_distribution.py
$SPARK_HOME/bin/spark-submit spark_lab/task2_sql_merchant_positive_ratio.py
$SPARK_HOME/bin/spark-submit spark_lab/task3_mllib_predict.py
```

也可以使用脚本批量运行：

```bash
bash spark_lab/run_all.sh
```

## 输出文件

关键输出文件（报告也会引用）：

- `outputs/task1_coupon_usage_counts_top10.txt`
- `outputs/task1_online_consumption_table_head10.txt`
- `outputs/task2_coupon_time_distribution.txt`
- `outputs/task2_top10_merchants_by_positive_ratio.txt`
- `outputs/task3_predictions.csv`
