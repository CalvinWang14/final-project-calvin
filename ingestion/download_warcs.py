import argparse
import io
import logging
import gzip

import boto3
import botocore
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType

logger = logging.getLogger(__name__)

CC_S3_BUCKET = "commoncrawl"

OUTPUT_SCHEMA = StructType([
    StructField("url", StringType(), True),
    StructField("crawl", StringType(), True),
    StructField("fetch_time", StringType(), True),
    StructField("html", StringType(), True),
    StructField("warc_filename", StringType(), True),
    StructField("warc_offset", LongType(), True),
])


def fetch_warc_record(warc_filename: str, offset: int, length: int) -> bytes:
    s3 = boto3.client(
        "s3",
        config=botocore.config.Config(
            retries={"max_attempts": 5, "mode": "adaptive"}
        ),
    )
    end = offset + length - 1
    resp = s3.get_object(
        Bucket=CC_S3_BUCKET,
        Key=warc_filename,
        Range=f"bytes={offset}-{end}",
    )
    raw = resp["Body"].read()
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass
    return raw


def parse_warc_record(raw: bytes):
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None

    parts = text.split("\r\n\r\n", 2)
    if len(parts) < 3:
        return None

    warc_header = parts[0]
    html = parts[2]

    uri = None
    for line in warc_header.splitlines():
        if line.lower().startswith("warc-target-uri:"):
            uri = line.split(":", 1)[1].strip()
            break

    if uri is None or not html.strip():
        return None

    return uri, html


def process_partition(rows):
    results = []
    for row in rows:
        try:
            raw = fetch_warc_record(
                row.warc_filename,
                int(row.warc_record_offset),
                int(row.warc_record_length),
            )
            parsed = parse_warc_record(raw)
            if parsed is None:
                continue
            uri, html = parsed
            results.append((
                uri,
                row.crawl,
                str(row.fetch_time),
                html[:500_000],
                row.warc_filename,
                int(row.warc_record_offset),
            ))
        except Exception as e:
            logger.warning(f"Error processing {row.url}: {e}")
            continue
    return iter(results)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--partitions", type=int, default=500)
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-WARCDownloader")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    logger.info(f"Reading manifest from {args.manifest}")
    manifest = spark.read.parquet(args.manifest)
    n = manifest.count()
    logger.info(f"Manifest size: {n:,} WARC records")

    n_partitions = max(args.partitions, n // 100)
    manifest_rdd = manifest.repartition(n_partitions).rdd

    logger.info(f"Fetching WARC records across {n_partitions} partitions...")
    results_rdd = manifest_rdd.mapPartitions(process_partition)

    results_df = spark.createDataFrame(results_rdd, schema=OUTPUT_SCHEMA)
    results_df = results_df.repartition(args.partitions)

    out = args.output.rstrip("/") + "/"
    logger.info(f"Writing raw HTML to {out}")
    results_df.write.mode("overwrite").parquet(out)

    logger.info(f"Done. Records written: {results_df.count():,}")
    spark.stop()


if __name__ == "__main__":
    main()
