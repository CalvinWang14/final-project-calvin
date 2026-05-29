# Political Polarization Across the Web Using Common Crawl

**Course:** MACS 30113

**Author:** Calvin Wang

---

## Research Problem

Political polarization in the United States has become one of the defining social science questions of the last two decades. Scholars have documented rising ideological divergence in Congress, declining cross-partisan friendship, and the geographic sorting of voters — but these studies typically rely on surveys, voting records, or manually curated media corpora. What is far harder to measure is the texture of everyday political discourse on the open web: the language ordinary political news outlets and blogs use, how it has shifted across years, and whether the online media ecosystem as a whole has become more or less fragmented along ideological lines.

This project uses **Common Crawl** — a petabyte-scale public archive of web crawls dating back to 2008 — to construct a large-scale longitudinal corpus of political web content and measure ideological polarization across domains and time. Specifically, the project asks:

1. **How have ideology scores and sentiment profiles of political news domains changed from 2022 to 2024?**
2. **Do political domains cluster into identifiable ideological groups, and how stable are these clusters across crawl snapshots?**
3. **Is aggregate polarization (inter-cluster ideological spread) increasing over time?**
4. **Which latent topics drive the strongest ideological differentiation across domains?**

These are substantive social science questions with real policy relevance. A systematic, data-driven answer requires processing hundreds of millions of web pages — a task that is simply not feasible without large-scale distributed computing.

---

## Why Scalable Computing Is Necessary

A single Common Crawl snapshot contains approximately **3 billion URLs** and roughly **400 terabytes** of raw compressed content. Even after filtering to political domains, a typical crawl yields tens of millions of pages. The analytical workflow has several stages that are individually and jointly computationally intensive:

**Data volume.** The Common Crawl columnar index alone — which I use to identify political-domain URLs before downloading WARC records — is stored as ~300 GB of Parquet files per crawl. Filtering this index sequentially on a single machine would take days; Spark distributes it across a cluster in minutes.

**WARC record fetching.** Each relevant page requires a byte-range HTTP request to the `commoncrawl` S3 bucket to retrieve its WARC record. Fetching tens of millions of records is an inherently parallel I/O problem; Spark's `mapPartitions` allows hundreds of concurrent fetch tasks across worker nodes.

**HTML parsing.** BeautifulSoup with lxml is a CPU-intensive operation. Parsing tens of millions of HTML documents serially would take weeks; distributed across 10 EMR nodes, it completes in hours.

**NLP at scale.** VADER sentiment analysis must run on every document (tens of millions). Spark's UDF mechanism broadcasts the VADER model to each worker, enabling row-level inference in parallel without driver-side bottlenecks.

**LDA topic modeling.** Spark MLlib's LDA implementation uses a distributed EM algorithm that scales to millions of documents and vocabularies of tens of thousands of terms. A corpus of this size cannot fit in the memory of a single machine.

**Temporal panel.** Analyzing six crawl snapshots across three years multiplies the data volume by 6x and requires merging outputs across crawls — another naturally parallel operation.

In short, every stage of this pipeline — from index filtering to topic modeling to clustering — operates at a scale that makes single-machine computing infeasible both in time and memory. The project is designed based on AWS EMR and PySpark.

---

## Data

**Source:** [Common Crawl](https://commoncrawl.org/) — a nonprofit that crawls the web monthly and provides the data for free on AWS S3.

**Crawl snapshots used:**
| Crawl ID | Approximate Date |
|---|---|
| CC-MAIN-2022-05 | January 2022 |
| CC-MAIN-2022-27 | July 2022 |
| CC-MAIN-2023-06 | February 2023 |
| CC-MAIN-2023-40 | October 2023 |
| CC-MAIN-2024-10 | March 2024 |
| CC-MAIN-2024-38 | September 2024 |

**Domain selection:** ~60 political news and opinion outlets spanning the full ideological spectrum (sourced from AllSides Media Bias Ratings and the Ad Fontes Media Bias Chart), supplemented by URL-path keyword filtering to capture political content on general news sites.

---

## Scalable Computing Methods

### Stage 1 — CC-Index Filtering (`ingestion/crawl_filter.py`)

The Common Crawl columnar index is stored in Parquet format at `s3://commoncrawl/cc-index/table/cc-main/warc/`. I read this ~300 GB dataset with Spark, filter to political domains using a domain allowlist and URL-path keyword list (applied as UDFs), and output a manifest of WARC record locations (filename, byte offset, length). This reduces hundreds of millions of records to hundreds of thousands.

### Stage 1b — WARC Record Downloading (`ingestion/download_warcs.py`)

Using the manifest from Stage 1, we fetch the raw WARC records from the `commoncrawl` S3 bucket via byte-range HTTP requests. Each Spark task handles a batch of ~100 records, and the work is distributed across hundreds of partitions. A minimal WARC parser extracts the HTML payload and target URL. Output is Parquet with columns `(url, html, crawl, fetch_time)`.

### Stage 2a — Text Extraction (`processing/extract_text.py`)

BeautifulSoup + lxml parses the raw HTML on each worker node, stripping boilerplate (navigation, footers, scripts) and extracting article-body text. A `min_word_count` threshold filters out thin pages. Spark UDFs apply extraction across all partitions in parallel.

### Stage 2b — Deduplication & Cleaning (`processing/clean_text.py`)

Near-duplicate pages (same URL canonical fingerprint, or identical first-500-character hash) are removed using Spark Window functions. A non-ASCII ratio filter removes non-English pages. Final text normalization (URL removal, lowercasing, whitespace collapse) prepares the corpus for NLP.

### Stage 3a — Sentiment Analysis (`analysis/sentiment_analysis.py`)

VADER (Valence Aware Dictionary and sEntiment Reasoner) computes compound, positive, negative, and neutral sentiment scores for each document. VADER is well-suited to short, informal web text and requires no GPU. We also compute a **partisan phrase score** — the normalized difference between left-coded and right-coded partisan phrase counts (based on Gentzkow & Shapiro 2010) — for each document. Results are aggregated to `(domain, year_month)` level.

### Stage 3b — Topic Modeling (`analysis/topic_modeling.py`)

Spark MLlib's distributed LDA (Latent Dirichlet Allocation) is fit on the full corpus using TF-IDF features (vocabulary of 20,000 terms, 30 topics, EM optimizer). The model outputs per-document topic distributions (a 30-dimensional vector) and per-topic term weights. Topic modeling at this scale — millions of documents, 20K vocabulary — requires distributed matrix factorization that MLlib handles via its block-partitioned EM implementation.

### Stage 3c — Ideology Scoring (`analysis/ideology_scoring.py`)

A **composite ideology score** is constructed for each document by combining:
1. The partisan phrase score from Stage 3a.
2. A topic-weighted ideology score: after manually labeling the 30 LDA topics on a [-1, +1] left-right scale (by inspecting top terms), each document's score is the probability-weighted average of its topic's ideology labels.

These are averaged into a single composite score per document, then aggregated to the `(domain, year_month)` level.

### Stage 4 — Clustering (`clustering/polarization_clusters.py`)

Domain-level ideology feature vectors (6 dimensions: ideology score, sentiment, partisan score, topic ideology, positive/negative sentiment) are assembled, standardized, and clustered using Spark MLlib K-Means. K is selected via Silhouette score across K ∈ {3, …, 8}. After collection to the driver, UMAP reduces the feature space to 2D for visualization. A **polarization index** is computed as the ratio of between-cluster to total variance in ideology scores — a measure analogous to the F-statistic in ANOVA.

### Stage 5 — Dashboard (`dashboard/app.py`)

A Streamlit dashboard with four views: (1) a UMAP scatter of domain ideology clusters, (2) time-series trend lines per domain, (3) an LDA topic explorer, and (4) the aggregate polarization index over time.

---

## Project Structure

```
political_polarization/
├── config.py                          # Central config (bucket, crawls, params)
├── requirements.txt
├── ingestion/
│   ├── crawl_filter.py                # Stage 1: Query CC-Index, emit WARC manifest
│   └── download_warcs.py             # Stage 1b: Fetch WARC records from S3
├── processing/
│   ├── extract_text.py               # Stage 2a: HTML → clean text (BS4 + Spark)
│   └── clean_text.py                 # Stage 2b: Dedup, language filter, normalize
├── analysis/
│   ├── sentiment_analysis.py         # Stage 3a: VADER + partisan phrase scoring
│   ├── topic_modeling.py             # Stage 3b: Distributed LDA (Spark MLlib)
│   └── ideology_scoring.py          # Stage 3c: Composite ideology score
├── clustering/
│   └── polarization_clusters.py     # Stage 4: K-Means + UMAP + polarization index
├── dashboard/
│   └── app.py                        # Stage 5: Streamlit dashboard
└── emr/
    ├── bootstrap.sh                  # EMR bootstrap (install Python deps)
    └── launch_cluster.py            # Launch EMR cluster + submit all steps
```

---

## Expected Findings

Based on prior literature and pilot analysis, we expect:

- Domains cluster into 4–5 stable ideological groups across crawls, with left- and right-leaning outlets showing wider lexical distance from center outlets than they do from each other (consistent with "affective polarization" findings in Iyengar et al. 2019).
- Sentiment negativity is higher at the ideological extremes, particularly for right-leaning outlets (consistent with Soroka et al. 2019 on negativity bias in partisan media).
- The polarization index increases monotonically from 2022 to 2024, with a detectable jump around the 2024 election cycle.
- Topics related to immigration, crime, and election integrity show the highest between-cluster ideological variance.

---

## References

- Gentzkow, M. & Shapiro, J. M. (2010). What Drives Media Slant? Evidence From U.S. Daily Newspapers. *Econometrica*, 78(1), 35–71.
- Iyengar, S., Lelkes, Y., Levendusky, M., Malhotra, N., & Westwood, S. J. (2019). The Origins and Consequences of Affective Polarization in the United States. *Annual Review of Political Science*, 22, 129–146.
- Soroka, S., Fournier, P., & Nir, L. (2019). Cross-national evidence of a negativity bias in psychophysiological reactions to news. *PNAS*, 116(38), 18888–18892.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022.
- Common Crawl. (2024). *Common Crawl Open Dataset*. https://commoncrawl.org/
