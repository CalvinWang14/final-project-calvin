import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, StringType

logger = logging.getLogger(__name__)

VADER_SCHEMA = StructType([
    StructField("compound", FloatType(), True),
    StructField("pos", FloatType(), True),
    StructField("neu", FloatType(), True),
    StructField("neg", FloatType(), True),
])


def vader_scores(text: str):
    if not text:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text[:2000])
        return (
            float(scores["compound"]),
            float(scores["pos"]),
            float(scores["neu"]),
            float(scores["neg"]),
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


vader_udf = F.udf(vader_scores, VADER_SCHEMA)

LEFT_PHRASES = [
    "climate change", "gun control", "reproductive rights", "income inequality",
    "universal healthcare", "systemic racism", "living wage", "affordable housing",
    "undocumented immigrants", "social justice", "police reform", "wealth tax",
    "green new deal", "medicaid expansion", "voting rights",
]

RIGHT_PHRASES = [
    "illegal aliens", "second amendment", "pro-life", "free market",
    "border security", "radical left", "cancel culture", "deep state",
    "election integrity", "school choice", "religious liberty", "energy independence",
    "traditional values", "law and order", "government overreach",
]


def partisan_score(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    left_count = sum(text_lower.count(p) for p in LEFT_PHRASES)
    right_count = sum(text_lower.count(p) for p in RIGHT_PHRASES)
    total = left_count + right_count
    if total == 0:
        return 0.0
    return float((right_count - left_count) / total)


partisan_udf = F.udf(partisan_score, FloatType())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-Sentiment")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    logger.info(f"Reading corpus from {args.input}")
    df = spark.read.parquet(args.input)

    scored = (
        df
        .withColumn("vader", vader_udf(F.col("text")))
        .withColumn("sentiment_compound", F.col("vader.compound"))
        .withColumn("sentiment_pos",      F.col("vader.pos"))
        .withColumn("sentiment_neg",      F.col("vader.neg"))
        .withColumn("sentiment_neu",      F.col("vader.neu"))
        .withColumn("partisan_score",     partisan_udf(F.col("text")))
        .drop("vader", "text")
    )

    out_doc = args.output.rstrip("/") + "/by_document/"
    logger.info(f"Writing per-document scores to {out_doc}")
    scored.repartition(200).write.mode("overwrite").parquet(out_doc)

    if args.aggregate:
        agg = (
            scored
            .groupBy("domain", "year_month")
            .agg(
                F.count("*").alias("doc_count"),
                F.avg("sentiment_compound").alias("avg_sentiment"),
                F.stddev("sentiment_compound").alias("std_sentiment"),
                F.avg("partisan_score").alias("avg_partisan_score"),
                F.stddev("partisan_score").alias("std_partisan_score"),
                F.avg("sentiment_pos").alias("avg_pos"),
                F.avg("sentiment_neg").alias("avg_neg"),
            )
            .filter(F.col("doc_count") >= 10)
            .orderBy("domain", "year_month")
        )

        out_agg = args.output.rstrip("/") + "/by_domain_month/"
        logger.info(f"Writing domain aggregates to {out_agg}")
        agg.repartition(50).write.mode("overwrite").parquet(out_agg)

    logger.info("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
