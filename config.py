S3_BUCKET = "polarization-project"
AWS_REGION = "us-east-1"

CRAWL_IDS = [
    "CC-MAIN-2022-05",
    "CC-MAIN-2022-27",
    "CC-MAIN-2023-06",
    "CC-MAIN-2023-40",
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-38",
]


def s3_path(stage: str, crawl: str = "") -> str:
    base = f"s3://{S3_BUCKET}/cc-political"
    if crawl:
        return f"{base}/{stage}/{crawl}/"
    return f"{base}/{stage}/"


POLITICAL_DOMAINS = [
    "huffpost.com", "motherjones.com", "thenation.com", "salon.com",
    "slate.com", "vox.com", "jacobin.com", "currentaffairs.org",
    "democracynow.org", "theintercept.com", "msnbc.com",
    "nytimes.com", "washingtonpost.com", "cnn.com", "theatlantic.com",
    "newyorker.com", "theguardian.com", "politico.com", "nbcnews.com",
    "apnews.com", "npr.org", "pbs.org",
    "reuters.com", "axios.com", "thehill.com", "csmonitor.com",
    "realclearpolitics.com", "militarytimes.com",
    "wsj.com", "reason.com", "nationalreview.com", "weeklystandard.com",
    "commentary.org", "city-journal.org",
    "foxnews.com", "nypost.com", "washingtonexaminer.com",
    "dailywire.com", "townhall.com", "powerlineblog.com",
    "breitbart.com", "thegatewaypundit.com", "infowars.com",
    "oann.com", "newsmax.com",
]

POLITICAL_KEYWORDS_URL = [
    "/politics/", "/election/", "/congress/", "/senate/",
    "/white-house/", "/government/", "/policy/",
    "/democrat/", "/republican/", "/liberal/", "/conservative/",
    "/immigration/", "/gun-control/", "/abortion/",
    "/supreme-court/", "/2024-election/", "/2022-midterms/",
    "/climate-policy/", "/healthcare-policy/",
]

LDA_NUM_TOPICS = 30
LDA_MAX_ITER = 20
VOCAB_SIZE = 20_000
MIN_WORD_COUNT = 100

KMEANS_K = 5
KMEANS_MAX_ITER = 100

EMR_CONFIG = {
    "master_instance_type": "m5.xlarge",
    "core_instance_type": "m5.2xlarge",
    "core_instance_count": 10,
    "release_label": "emr-6.15.0",
    "applications": ["Spark", "Hadoop"],
}
