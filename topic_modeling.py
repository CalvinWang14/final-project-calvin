import argparse
import logging
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
)
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline

logger = logging.getLogger(__name__)

POLITICAL_STOP_WORDS = [
    "said", "says", "would", "could", "also", "new", "one", "two",
    "year", "years", "time", "day", "people", "us", "like", "just",
    "get", "make", "way", "mr", "ms", "mrs", "dr", "com", "www",
    "click", "subscribe", "newsletter", "advertisement", "sponsored",
    "read", "more", "share", "tweet", "facebook", "twitter",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-topics", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--vocab-size", type=int, default=20_000)
    p.add_argument("--min-df", type=int, default=5)
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-TopicModeling")
        .config("spark.sql.shuffle.partitions", "300")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "12g")
        .getOrCreate()
    )


def extract_top_terms(vocab, topics_matrix, n_top=15):
    top_terms = {}
    for topic_idx, topic_vec in enumerate(topics_matrix):
        sorted_indices = sorted(
            range(len(topic_vec)), key=lambda i: topic_vec[i], reverse=True
        )[:n_top]
        top_terms[topic_idx] = [vocab[i] for i in sorted_indices]
    return top_terms


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    logger.info(f"Reading corpus from {args.input}")
    df = spark.read.parquet(args.input).select("url", "domain", "year_month", "text")

    df = df.withColumn("doc_id", F.monotonically_increasing_id())
    df.cache()
    n = df.count()
    logger.info(f"Corpus size: {n:,} documents")

    tokenizer = RegexTokenizer(
        inputCol="text", outputCol="tokens",
        pattern=r"[^a-z]+", toLowercase=True, minTokenLength=3
    )

    stop_words = StopWordsRemover.loadDefaultStopWords("english") + POLITICAL_STOP_WORDS
    remover = StopWordsRemover(
        inputCol="tokens", outputCol="filtered_tokens",
        stopWords=stop_words
    )

    cv = CountVectorizer(
        inputCol="filtered_tokens", outputCol="raw_features",
        vocabSize=args.vocab_size, minDF=args.min_df
    )

    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=args.min_df)

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])

    logger.info("Fitting TF-IDF pipeline...")
    pipeline_model = pipeline.fit(df)
    featurized = pipeline_model.transform(df)

    vocab = pipeline_model.stages[2].vocabulary
    logger.info(f"Vocabulary size: {len(vocab):,}")

    lda = LDA(
        k=args.num_topics,
        maxIter=args.max_iter,
        optimizer="em",
        featuresCol="features",
        topicDistributionCol="topic_dist",
        docConcentration=[1.0 / args.num_topics] * args.num_topics,
        topicConcentration=1.0 / args.num_topics,
    )

    logger.info(f"Fitting LDA with k={args.num_topics}, maxIter={args.max_iter}...")
    lda_model = lda.fit(featurized)

    ll = lda_model.logLikelihood(featurized)
    perplexity = lda_model.logPerplexity(featurized)
    logger.info(f"LDA log-likelihood: {ll:.2f}, perplexity: {perplexity:.4f}")

    topics_matrix = lda_model.topicsMatrix()
    top_terms = extract_top_terms(vocab, topics_matrix.toArray().T, n_top=20)

    topics_json = json.dumps(top_terms, indent=2)
    sc.parallelize([topics_json], 1).saveAsTextFile(
        args.output.rstrip("/") + "/topic_terms/"
    )
    logger.info("Top terms per topic saved.")

    transformed = lda_model.transform(featurized)

    to_array = F.udf(lambda v: v.toArray().tolist() if v else None,
                     ArrayType(FloatType()))

    doc_topics = (
        transformed
        .withColumn("topic_dist_arr", to_array(F.col("topic_dist")))
        .withColumn("dominant_topic",
                    F.array_position(
                        F.col("topic_dist_arr"),
                        F.array_max(F.col("topic_dist_arr"))
                    ).cast("int") - 1)
        .select("doc_id", "url", "domain", "year_month",
                "topic_dist_arr", "dominant_topic")
    )

    out_docs = args.output.rstrip("/") + "/doc_topics/"
    logger.info(f"Writing doc-topic distributions to {out_docs}")
    doc_topics.repartition(200).write.mode("overwrite").parquet(out_docs)

    model_path = args.output.rstrip("/") + "/model/"
    logger.info(f"Saving LDA model to {model_path}")
    lda_model.save(model_path)

    logger.info("Topic modeling complete.")
    spark.stop()


if __name__ == "__main__":
    main()
