import argparse
import logging
import re
import unicodedata

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField

logger = logging.getLogger(__name__)

MIN_WORD_COUNT = 100
MAX_CHARS = 20_000


def extract_text_from_html(html: str) -> str | None:
    if not html:
        return None
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe", "ads",
                          "advertisement"]):
            tag.decompose()

        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"(article|content|post|story|body)", re.I))
            or soup.body
        )

        if article is None:
            return None

        lines = []
        for elem in article.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote"]):
            t = elem.get_text(separator=" ", strip=True)
            if t:
                lines.append(t)

        text = " ".join(lines)
        text = normalize_text(text)

        if len(text.split()) < MIN_WORD_COUNT:
            return None

        return text[:MAX_CHARS]

    except Exception as e:
        logger.debug(f"BS4 extraction failed: {e}")
        return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


extract_text_udf = F.udf(extract_text_from_html, StringType())


def extract_domain(url: str) -> str | None:
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().lstrip("www.")
        return host if host else None
    except Exception:
        return None


extract_domain_udf = F.udf(extract_domain, StringType())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--crawl", required=True)
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-ExtractText")
        .config("spark.sql.shuffle.partitions", "300")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    logger.info(f"Reading raw HTML from {args.input}")
    raw = spark.read.parquet(args.input)

    processed = (
        raw
        .withColumn("text", extract_text_udf(F.col("html")))
        .filter(F.col("text").isNotNull())
        .withColumn("domain", extract_domain_udf(F.col("url")))
        .filter(F.col("domain").isNotNull())
        .withColumn("word_count", F.size(F.split(F.col("text"), r"\s+")))
        .filter(F.col("word_count") >= MIN_WORD_COUNT)
        .select(
            "url",
            "domain",
            "crawl",
            "fetch_time",
            "text",
            "word_count",
        )
    )

    n = processed.count()
    logger.info(f"Documents after extraction: {n:,}")

    out = args.output.rstrip("/") + "/"
    processed.repartition(200).write.mode("overwrite").parquet(out)
    logger.info(f"Written to {out}")
    spark.stop()


if __name__ == "__main__":
    main()
