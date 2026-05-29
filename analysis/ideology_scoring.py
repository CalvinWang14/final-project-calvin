import argparse
import json
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ArrayType

logger = logging.getLogger(__name__)


def load_topic_labels(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def make_ideology_udf(topic_labels: dict):
    def _score(topic_dist_arr):
        if topic_dist_arr is None:
            return 0.0
        total = sum(topic_dist_arr)
        if total == 0:
            return 0.0
        score = sum(
            topic_dist_arr[i] * topic_labels.get(i, 0.0)
            for i in range(len(topic_dist_arr))
        )
        return float(score / total)

    return F.udf(_score, FloatType())


def build_panel(spark, base_s3_path: str, crawl_ids: list) -> "DataFrame":
    dfs = []
    for crawl in crawl_ids:
        path = f"{base_s3_path.rstrip('/')}/{crawl}/by_domain_month/"
        try:
            dfs.append(spark.read.parquet(path).withColumn("crawl", F.lit(crawl)))
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    if not dfs:
        raise ValueError("No crawl data found.")
    return dfs[0].unionByName(*dfs[1:]) if len(dfs) > 1 else dfs[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sentiment", required=True)
    p.add_argument("--topics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topic-labels", required=True)
    p.add_argument("--crawls", nargs="+")
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-IdeologyScoring")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    topic_labels = load_topic_labels(args.topic_labels)
    logger.info(f"Loaded ideology labels for {len(topic_labels)} topics")

    ideology_udf = make_ideology_udf(topic_labels)

    doc_topics_path = args.topics.rstrip("/") + "/doc_topics/"
    logger.info(f"Reading doc topics from {doc_topics_path}")
    doc_topics = spark.read.parquet(doc_topics_path)

    doc_ideology = (
        doc_topics
        .withColumn("topic_ideology_score", ideology_udf(F.col("topic_dist_arr")))
        .select("url", "domain", "year_month", "dominant_topic",
                "topic_ideology_score")
    )

    sent_doc_path = args.sentiment.rstrip("/") + "/by_document/"
    sent_docs = spark.read.parquet(sent_doc_path).select(
        "url", "sentiment_compound", "partisan_score",
        "sentiment_pos", "sentiment_neg"
    )

    joined = doc_ideology.join(sent_docs, on="url", how="inner")

    joined = joined.withColumn(
        "composite_ideology",
        (F.col("partisan_score") + F.col("topic_ideology_score")) / 2.0
    )

    domain_agg = (
        joined
        .groupBy("domain", "year_month")
        .agg(
            F.count("*").alias("doc_count"),
            F.avg("composite_ideology").alias("ideology_score"),
            F.stddev("composite_ideology").alias("ideology_std"),
            F.avg("sentiment_compound").alias("avg_sentiment"),
            F.avg("sentiment_pos").alias("avg_pos_sentiment"),
            F.avg("sentiment_neg").alias("avg_neg_sentiment"),
            F.avg("topic_ideology_score").alias("avg_topic_ideology"),
            F.avg("partisan_score").alias("avg_partisan_score"),
            F.expr("percentile_approx(dominant_topic, 0.5)").cast("int")
              .alias("modal_topic"),
        )
        .filter(F.col("doc_count") >= 5)
        .orderBy("domain", "year_month")
    )

    out_agg = args.output.rstrip("/") + "/domain_ideology/"
    logger.info(f"Writing domain ideology to {out_agg}")
    domain_agg.repartition(50).write.mode("overwrite").parquet(out_agg)

    out_doc = args.output.rstrip("/") + "/doc_ideology/"
    logger.info(f"Writing doc-level ideology to {out_doc}")
    joined.select(
        "url", "domain", "year_month", "dominant_topic",
        "composite_ideology", "sentiment_compound",
        "partisan_score", "topic_ideology_score"
    ).repartition(200).write.mode("overwrite").parquet(out_doc)

    logger.info("Ideology scoring complete.")
    spark.stop()


if __name__ == "__main__":
    main()
