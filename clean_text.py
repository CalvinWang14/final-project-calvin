import argparse
import hashlib
import logging
import re
import unicodedata

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType

logger = logging.getLogger(__name__)

MAX_NON_ASCII_RATIO = 0.15


def url_fingerprint(url: str) -> str | None:
    if not url:
        return None
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url.lower())
        path = re.sub(r"/+$", "", p.path)
        canonical = urlunparse((p.scheme.replace("http", ""),
                                p.netloc, path, "", "", ""))
        return hashlib.md5(canonical.encode()).hexdigest()
    except Exception:
        return None


def text_fingerprint(text: str) -> str | None:
    if not text:
        return None
    snippet = " ".join(text[:500].lower().split())
    return hashlib.md5(snippet.encode()).hexdigest()


url_fp_udf = F.udf(url_fingerprint, StringType())
text_fp_udf = F.udf(text_fingerprint, StringType())


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1)


non_ascii_udf = F.udf(non_ascii_ratio, FloatType())


def clean_for_nlp(text: str) -> str | None:
    if not text:
        return None
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text.split()) >= 50 else None


clean_udf = F.udf(clean_for_nlp, StringType())


def crawl_to_year_month(crawl_id: str) -> str | None:
    if not crawl_id:
        return None
    m = re.search(r"(\d{4})-(\d{2})", crawl_id)
    if m:
        year = int(m.group(1))
        week = int(m.group(2))
        from datetime import date, timedelta
        d = date.fromisocalendar(year, min(week, 52), 1)
        return d.strftime("%Y-%m")
    return None


crawl_date_udf = F.udf(crawl_to_year_month, StringType())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-CleanText")
        .config("spark.sql.shuffle.partitions", "300")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    logger.info(f"Reading from {args.input}")
    df = spark.read.parquet(args.input)

    df = (
        df
        .withColumn("url_fp", url_fp_udf(F.col("url")))
        .withColumn("text_fp", text_fp_udf(F.col("text")))
    )

    w_url = Window.partitionBy("url_fp").orderBy("fetch_time")
    df = (
        df
        .withColumn("rn_url", F.row_number().over(w_url))
        .filter(F.col("rn_url") == 1)
        .drop("rn_url")
    )

    w_text = Window.partitionBy("text_fp").orderBy("fetch_time")
    df = (
        df
        .withColumn("rn_text", F.row_number().over(w_text))
        .filter(F.col("rn_text") == 1)
        .drop("rn_text")
    )

    df = (
        df
        .withColumn("non_ascii_ratio", non_ascii_udf(F.col("text")))
        .filter(F.col("non_ascii_ratio") <= MAX_NON_ASCII_RATIO)
        .drop("non_ascii_ratio")
    )

    df = (
        df
        .withColumn("clean_text", clean_udf(F.col("text")))
        .filter(F.col("clean_text").isNotNull())
        .withColumn("year_month", crawl_date_udf(F.col("crawl")))
        .drop("text", "url_fp", "text_fp")
        .withColumnRenamed("clean_text", "text")
    )

    n = df.count()
    logger.info(f"Final corpus size: {n:,}")

    out = args.output.rstrip("/") + "/"
    df.repartition(200).write.mode("overwrite").parquet(out)
    logger.info(f"Written to {out}")
    spark.stop()


if __name__ == "__main__":
    main()
