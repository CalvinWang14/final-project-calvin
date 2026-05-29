import argparse
import logging

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ideology_score",
    "avg_sentiment",
    "avg_pos_sentiment",
    "avg_neg_sentiment",
    "avg_partisan_score",
    "avg_topic_ideology",
]


def select_k(df_features, k_range=(3, 8)) -> int:
    best_k, best_score = k_range[0], -1.0
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    for k in range(*k_range):
        km = KMeans(
            k=k, seed=42,
            featuresCol="scaled_features",
            predictionCol="cluster",
            maxIter=50,
            tol=1e-4,
        )
        model = km.fit(df_features)
        preds = model.transform(df_features)
        score = evaluator.evaluate(preds)
        logger.info(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    logger.info(f"Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k


def compute_umap(feature_matrix: np.ndarray, n_neighbors=15,
                 min_dist=0.1) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            metric="euclidean",
        )
        return reducer.fit_transform(feature_matrix)
    except ImportError:
        logger.warning("umap-learn not installed; falling back to PCA for 2D projection")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(feature_matrix)


def compute_polarization_index(cluster_assignments: pd.DataFrame) -> dict:
    clusters = cluster_assignments["cluster"].unique()
    global_mean = cluster_assignments["ideology_score"].mean()

    cluster_means = cluster_assignments.groupby("cluster")["ideology_score"].mean()
    cluster_sizes = cluster_assignments.groupby("cluster").size()
    between_var = sum(
        cluster_sizes[c] * (cluster_means[c] - global_mean) ** 2
        for c in clusters
    ) / len(cluster_assignments)

    within_var = cluster_assignments.groupby("cluster")["ideology_score"].var()
    within_var_weighted = sum(
        cluster_sizes[c] * within_var.get(c, 0.0)
        for c in clusters
    ) / len(cluster_assignments)

    total_var = between_var + within_var_weighted
    pi = between_var / total_var if total_var > 0 else 0.0

    return {
        "polarization_index": round(pi, 4),
        "between_cluster_var": round(between_var, 4),
        "within_cluster_var": round(within_var_weighted, 4),
        "n_clusters": len(clusters),
        "n_domains": len(cluster_assignments),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--auto-k-max", type=int, default=8)
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("CC-Clustering")
        .config("spark.sql.shuffle.partitions", "100")
        .getOrCreate()
    )


def main():
    args = parse_args()
    spark = build_spark()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    logger.info(f"Reading ideology features from {args.input}")
    df = spark.read.parquet(args.input)

    domain_features = (
        df
        .groupBy("domain")
        .agg(
            *[F.avg(c).alias(c) for c in FEATURE_COLS],
            F.sum("doc_count").alias("total_docs"),
        )
        .filter(F.col("total_docs") >= 20)
    )

    n_domains = domain_features.count()
    logger.info(f"Domains for clustering: {n_domains:,}")

    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=True, withStd=True,
    )

    assembled = assembler.transform(domain_features)
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)

    k = args.k
    if k <= 0:
        logger.info(f"Auto-selecting K in range [3, {args.auto_k_max})...")
        k = select_k(scaled, k_range=(3, args.auto_k_max))

    km = KMeans(
        k=k, seed=42,
        featuresCol="scaled_features",
        predictionCol="cluster",
        maxIter=100, tol=1e-6,
    )
    logger.info(f"Fitting K-Means with k={k}...")
    km_model = km.fit(scaled)

    clustered = km_model.transform(scaled)

    cols_to_collect = ["domain", "cluster", "total_docs"] + FEATURE_COLS
    pdf = clustered.select(cols_to_collect).toPandas()
    logger.info(f"Collected {len(pdf)} domain rows to driver")

    feature_matrix = pdf[FEATURE_COLS].fillna(0).values
    umap_coords = compute_umap(feature_matrix)
    pdf["umap_x"] = umap_coords[:, 0]
    pdf["umap_y"] = umap_coords[:, 1]

    pi_metrics = compute_polarization_index(pdf)
    logger.info(f"Polarization metrics: {pi_metrics}")

    cluster_summary = (
        pdf.groupby("cluster")
        .agg(
            ideology_score_mean=("ideology_score", "mean"),
            avg_sentiment_mean=("avg_sentiment", "mean"),
            doc_count_sum=("total_docs", "sum"),
            domain_count=("domain", "count"),
        )
        .reset_index()
        .sort_values("ideology_score_mean")
    )

    ideology_labels = {
        i: label for i, (_, label) in enumerate(
            zip(cluster_summary.itertuples(),
                _assign_labels(cluster_summary["ideology_score_mean"].values, k))
        )
    }
    cluster_summary["ideology_label"] = cluster_summary["cluster"].map(ideology_labels)
    pdf["ideology_label"] = pdf["cluster"].map(ideology_labels)

    out_base = args.output.rstrip("/")

    results_df = spark.createDataFrame(pdf)
    results_df.repartition(20).write.mode("overwrite").parquet(
        f"{out_base}/domain_clusters/"
    )
    logger.info(f"Domain clusters written to {out_base}/domain_clusters/")

    sc.parallelize(
        cluster_summary.to_csv(index=False).splitlines(), 1
    ).saveAsTextFile(f"{out_base}/cluster_summary/")

    import json
    pi_json = json.dumps({**pi_metrics, "k": k})
    sc.parallelize([pi_json], 1).saveAsTextFile(
        f"{out_base}/polarization_metrics/"
    )

    logger.info("Clustering complete.")
    spark.stop()


def _assign_labels(ideology_means: np.ndarray, k: int) -> list:
    n = len(ideology_means)
    sorted_idx = np.argsort(ideology_means)
    labels = [""] * n
    if k <= 3:
        label_map = {0: "Left", 1: "Center", 2: "Right"}
    else:
        label_map = {
            0: "Far Left", 1: "Center-Left", 2: "Center",
            3: "Center-Right", 4: "Far Right",
        }
        for i in range(5, k):
            label_map[i] = f"Cluster {i}"
    for rank, idx in enumerate(sorted_idx):
        labels[idx] = label_map.get(rank, f"Cluster {rank}")
    return labels


if __name__ == "__main__":
    main()
