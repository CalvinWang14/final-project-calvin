"""
app.py
------
Streamlit dashboard for the Political Polarization project.
Reads processed Parquet/CSV outputs from S3 (or local disk) and renders:

  Tab 1 — Ideology Map:     UMAP 2D scatter of domains colored by cluster
  Tab 2 — Trend Lines:      Time-series of ideology score + sentiment by domain
  Tab 3 — Topic Explorer:   Topic prevalence over time, top terms per topic
  Tab 4 — Polarization:     Aggregate polarization index over crawl history

Run locally:
    streamlit run app.py -- --data-dir ./data/

Run pointing at S3:
    streamlit run app.py -- --data-dir s3://YOUR-BUCKET/cc-political/
"""

import argparse
import json
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Argument parsing (Streamlit passes args after --)
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data",
                        help="Base directory with pipeline outputs")
    parser.add_argument("--crawls", nargs="+",
                        default=["CC-MAIN-2022-05", "CC-MAIN-2022-27",
                                 "CC-MAIN-2023-06", "CC-MAIN-2023-40",
                                 "CC-MAIN-2024-10", "CC-MAIN-2024-38"])
    try:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:]
                                 if "--" in sys.argv else [])
    except SystemExit:
        args = parser.parse_args([])
    return args


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_domain_clusters(data_dir: str, crawl: str) -> pd.DataFrame:
    path = f"{data_dir.rstrip('/')}/clusters/{crawl}/domain_clusters/"
    try:
        return pd.read_parquet(path)
    except Exception:
        # Fallback: generate synthetic data for demo
        np.random.seed(42)
        n = 80
        return pd.DataFrame({
            "domain": [f"site{i}.com" for i in range(n)],
            "cluster": np.random.randint(0, 5, n),
            "ideology_label": np.random.choice(
                ["Far Left", "Center-Left", "Center", "Center-Right", "Far Right"], n),
            "ideology_score": np.random.uniform(-1, 1, n),
            "avg_sentiment": np.random.uniform(-0.5, 0.5, n),
            "umap_x": np.random.randn(n),
            "umap_y": np.random.randn(n),
            "total_docs": np.random.randint(20, 5000, n),
        })


@st.cache_data(ttl=3600)
def load_domain_ideology_timeseries(data_dir: str) -> pd.DataFrame:
    """Load and concatenate ideology time-series across all available crawls."""
    dfs = []
    import os, glob
    pattern = f"{data_dir.rstrip('/')}/ideology/*/domain_ideology/"
    for path in glob.glob(pattern):
        try:
            df = pd.read_parquet(path)
            dfs.append(df)
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    # Synthetic fallback
    np.random.seed(0)
    domains = ["foxnews.com", "cnn.com", "nytimes.com",
               "breitbart.com", "msnbc.com", "reuters.com"]
    months = pd.date_range("2022-01", "2024-12", freq="MS").strftime("%Y-%m")
    rows = []
    ideology_baselines = {"foxnews.com": 0.7, "cnn.com": -0.4,
                          "nytimes.com": -0.3, "breitbart.com": 0.9,
                          "msnbc.com": -0.6, "reuters.com": 0.0}
    for d in domains:
        for m in months:
            rows.append({
                "domain": d, "year_month": m,
                "ideology_score": ideology_baselines[d] + np.random.normal(0, 0.08),
                "avg_sentiment": np.random.uniform(-0.3, 0.3),
                "avg_partisan_score": ideology_baselines[d] + np.random.normal(0, 0.1),
                "doc_count": np.random.randint(50, 2000),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_topic_terms(data_dir: str, crawl: str) -> dict:
    path = f"{data_dir.rstrip('/')}/topics/{crawl}/topic_terms/topics.json"
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {str(i): [f"word_{i}_{j}" for j in range(15)] for i in range(10)}


@st.cache_data(ttl=3600)
def load_polarization_metrics(data_dir: str) -> pd.DataFrame:
    """Load polarization index per crawl."""
    import os, glob
    rows = []
    for path in glob.glob(f"{data_dir.rstrip('/')}/clusters/*/polarization_metrics/"):
        try:
            files = glob.glob(f"{path}part-*")
            if files:
                with open(files[0]) as f:
                    data = json.load(f)
                crawl = path.split("/")[-3]
                rows.append({"crawl": crawl, **data})
        except Exception:
            pass
    if rows:
        return pd.DataFrame(rows)
    # Synthetic fallback
    crawls = ["CC-MAIN-2022-05", "CC-MAIN-2022-27",
              "CC-MAIN-2023-06", "CC-MAIN-2023-40",
              "CC-MAIN-2024-10", "CC-MAIN-2024-38"]
    return pd.DataFrame({
        "crawl": crawls,
        "polarization_index": [0.42, 0.44, 0.47, 0.51, 0.55, 0.58],
        "between_cluster_var": [0.12, 0.13, 0.14, 0.16, 0.18, 0.20],
        "within_cluster_var": [0.17, 0.16, 0.15, 0.15, 0.15, 0.15],
    })


# ---------------------------------------------------------------------------
# Color palette for ideology labels
# ---------------------------------------------------------------------------
IDEOLOGY_COLORS = {
    "Far Left":      "#1565C0",
    "Center-Left":   "#64B5F6",
    "Center":        "#9E9E9E",
    "Center-Right":  "#EF9A9A",
    "Far Right":     "#B71C1C",
}


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    data_dir = args.data_dir

    st.set_page_config(
        page_title="Political Polarization Across the Web",
        page_icon="🗳️",
        layout="wide",
    )

    st.title("🗳️ Political Polarization Across the Web")
    st.markdown(
        "**Source:** Common Crawl | "
        "**Pipeline:** PySpark on AWS EMR | "
        "**Analysis:** VADER Sentiment · LDA Topic Modeling · K-Means Clustering"
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        available_crawls = args.crawls
        selected_crawl = st.selectbox("Select crawl snapshot", available_crawls,
                                      index=len(available_crawls) - 1)
        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "This dashboard visualizes political ideology and sentiment signals "
            "extracted from millions of political web pages collected by Common Crawl."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Ideology Map",
        "📈 Trend Lines",
        "💬 Topic Explorer",
        "📊 Polarization Index",
    ])

    # -------------------------------------------------------------------
    # Tab 1: Ideology Map (UMAP scatter)
    # -------------------------------------------------------------------
    with tab1:
        st.subheader("Domain Ideology Map")
        st.markdown(
            "Each point is a news domain. Position reflects ideology (left/right) "
            "and sentiment profile. Color = cluster assignment."
        )

        clusters_df = load_domain_clusters(data_dir, selected_crawl)

        col1, col2 = st.columns([3, 1])
        with col2:
            min_docs = st.slider("Min documents", 10, 1000, 50, step=10)
            show_labels = st.checkbox("Show domain labels", value=False)

        filtered = clusters_df[clusters_df["total_docs"] >= min_docs]

        fig = px.scatter(
            filtered,
            x="umap_x", y="umap_y",
            color="ideology_label",
            color_discrete_map=IDEOLOGY_COLORS,
            size="total_docs",
            size_max=30,
            hover_data=["domain", "ideology_score", "avg_sentiment", "total_docs"],
            text="domain" if show_labels else None,
            title=f"Domain Ideology Clusters — {selected_crawl}",
            labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
            height=600,
        )
        fig.update_traces(marker=dict(opacity=0.8))
        if show_labels:
            fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(
            legend_title="Ideology Cluster",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        # Cluster summary table
        st.subheader("Cluster Summary")
        summary = (
            filtered.groupby("ideology_label")
            .agg(
                Domains=("domain", "count"),
                Avg_Ideology=("ideology_score", "mean"),
                Avg_Sentiment=("avg_sentiment", "mean"),
                Total_Docs=("total_docs", "sum"),
            )
            .reset_index()
            .rename(columns={"ideology_label": "Cluster"})
            .sort_values("Avg_Ideology")
        )
        st.dataframe(
            summary.style.format({
                "Avg_Ideology": "{:.3f}",
                "Avg_Sentiment": "{:.3f}",
                "Total_Docs": "{:,}",
            }),
            use_container_width=True,
        )

    # -------------------------------------------------------------------
    # Tab 2: Trend Lines
    # -------------------------------------------------------------------
    with tab2:
        st.subheader("Ideology & Sentiment Over Time")

        ts_df = load_domain_ideology_timeseries(data_dir)

        col1, col2 = st.columns(2)
        with col1:
            all_domains = sorted(ts_df["domain"].unique().tolist())
            default_domains = all_domains[:6] if len(all_domains) >= 6 else all_domains
            selected_domains = st.multiselect(
                "Select domains", all_domains, default=default_domains
            )
        with col2:
            metric = st.selectbox(
                "Metric",
                ["ideology_score", "avg_sentiment", "avg_partisan_score"],
                format_func=lambda x: {
                    "ideology_score": "Composite Ideology Score",
                    "avg_sentiment": "VADER Sentiment (compound)",
                    "avg_partisan_score": "Partisan Phrase Score",
                }[x],
            )

        if selected_domains:
            plot_df = ts_df[ts_df["domain"].isin(selected_domains)].copy()
            plot_df = plot_df.sort_values("year_month")

            fig2 = px.line(
                plot_df,
                x="year_month", y=metric,
                color="domain",
                markers=True,
                title=f"{metric.replace('_', ' ').title()} Over Time",
                labels={"year_month": "Month", metric: metric.replace("_", " ").title()},
                height=500,
            )
            fig2.add_hline(y=0, line_dash="dot", line_color="gray",
                           annotation_text="Neutral (0)")
            fig2.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[-1.1, 1.1]),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Select at least one domain above.")

    # -------------------------------------------------------------------
    # Tab 3: Topic Explorer
    # -------------------------------------------------------------------
    with tab3:
        st.subheader("LDA Topic Explorer")
        st.markdown(
            "The topic model identified 30 latent topics across the political corpus. "
            "Select a topic to see its top terms."
        )

        topics = load_topic_terms(data_dir, selected_crawl)

        col1, col2 = st.columns([1, 2])
        with col1:
            topic_id = st.selectbox(
                "Topic", list(topics.keys()),
                format_func=lambda x: f"Topic {x}"
            )
        with col2:
            if topic_id in topics:
                terms = topics[topic_id][:20]
                # Bar chart of pseudo-weights (rank-based)
                weights = [1 / (i + 1) for i in range(len(terms))]
                fig3 = px.bar(
                    x=weights, y=terms,
                    orientation="h",
                    labels={"x": "Relative Weight", "y": "Term"},
                    title=f"Top Terms — Topic {topic_id}",
                    color=weights,
                    color_continuous_scale="Blues",
                    height=500,
                )
                fig3.update_layout(
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------------------------------------------
    # Tab 4: Polarization Index
    # -------------------------------------------------------------------
    with tab4:
        st.subheader("Aggregate Polarization Over Time")
        st.markdown(
            "The **Polarization Index** measures how much inter-cluster variance "
            "dominates over intra-cluster variance in ideology scores. "
            "Higher = more polarized media landscape."
        )

        pi_df = load_polarization_metrics(data_dir)
        pi_df = pi_df.sort_values("crawl")

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=pi_df["crawl"], y=pi_df["polarization_index"],
            mode="lines+markers", name="Polarization Index",
            line=dict(color="#E53935", width=2.5),
            marker=dict(size=8),
        ))
        fig4.add_trace(go.Scatter(
            x=pi_df["crawl"], y=pi_df["between_cluster_var"],
            mode="lines+markers", name="Between-Cluster Variance",
            line=dict(color="#1565C0", width=2, dash="dash"),
        ))
        fig4.add_trace(go.Scatter(
            x=pi_df["crawl"], y=pi_df["within_cluster_var"],
            mode="lines+markers", name="Within-Cluster Variance",
            line=dict(color="#2E7D32", width=2, dash="dot"),
        ))
        fig4.update_layout(
            title="Political Polarization Index Across Common Crawl Snapshots",
            xaxis_title="Crawl Snapshot",
            yaxis_title="Value",
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Metric cards
        if len(pi_df) >= 2:
            latest = pi_df.iloc[-1]
            earliest = pi_df.iloc[0]
            delta = latest["polarization_index"] - earliest["polarization_index"]

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Latest Polarization Index",
                f"{latest['polarization_index']:.3f}",
                delta=f"{delta:+.3f} since {earliest['crawl']}",
                delta_color="inverse",
            )
            col2.metric(
                "Between-Cluster Variance",
                f"{latest['between_cluster_var']:.3f}",
            )
            col3.metric(
                "Within-Cluster Variance",
                f"{latest['within_cluster_var']:.3f}",
            )


if __name__ == "__main__":
    main()
