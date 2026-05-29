import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from config import POLITICAL_DOMAINS, POLITICAL_KEYWORDS_URL, S3_BUCKET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN_ALLOWLIST = set(POLITICAL_DOMAINS)
URL_PATH_KEYWORDS = POLITICAL_KEYWORDS_URL


def is_political_domain(url: str) -> bool:
    if url is None:
        return False
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in DOMAIN_ALLOWLIST)
    except Exception:
        return False


def is_political_url_path(url: str) -> bool:
    if url is None:
        return False
    url_lower = url.lower()
    return any(kw in url_lower for kw in URL_PATH_KEYWORDS)


is_political_domain_udf = F.udf(is_political_domain, StringType())
is_political_url_udf = F.udf(is_political_url_path, StringType())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crawl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-fraction", type=float, default=1.0)
    return p.parse_args()


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("CC-PoliticalFilter")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    index_path = f"s3://commoncrawl/cc-index/table/cc-main/warc/"

    logger.info(f"Reading CC index from {index_path}")
    df = spark.read.parquet(index_path)
    df = df.filter(F.col("crawl") == args.crawl)

    if args.sample_fraction < 1.0:
        df = df.sample(fraction=args.sample_fraction, seed=42)

    logger.info(f"Total records in crawl {args.crawl}: {df.count():,}")

    spark.udf.register("is_political_domain", is_political_domain, StringType())
    spark.udf.register("is_political_url_path", is_political_url_path, StringType())

    political_df = df.filter(
        (is_political_domain_udf(F.col("url")) == "True") |
        (is_political_url_udf(F.col("url")) == "True")
    ).select(
        "url",
        "fetch_time",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
        "content_languages",
        "content_mime_detected",
        "crawl",
    ).filter(
        (F.col("content_languages").contains("eng")) &
        (F.col("content_mime_detected") == "text/html")
    )

    n_political = political_df.count()
    logger.info(f"Political records identified: {n_political:,}")

    political_df = political_df.repartition(200)

    out = f"{args.output.rstrip('/')}/{args.crawl}/"
    logger.info(f"Writing segment manifest to {out}")
    political_df.write.mode("overwrite").parquet(out)

    logger.info("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
