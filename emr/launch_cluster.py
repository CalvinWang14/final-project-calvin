import argparse
import os
import boto3

REGION = "us-east-1"
EMR_RELEASE = "emr-6.15.0"
MASTER_INSTANCE = "m5.xlarge"
CORE_INSTANCE   = "m5.2xlarge"
CORE_COUNT      = 4


def upload_scripts(s3_client, bucket: str, local_dir: str, s3_prefix: str):
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            if fname.endswith(".py") or fname.endswith(".sh"):
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{rel}".replace("\\", "/")
                print(f"  Uploading {local_path} -> s3://{bucket}/{s3_key}")
                s3_client.upload_file(local_path, bucket, s3_key)


def make_spark_step(name: str, script_s3: str, args: list, py_files: str) -> dict:
    return {
        "Name": name,
        "ActionOnFailure": "CONTINUE",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--master", "yarn",
                "--deploy-mode", "cluster",
                "--conf", "spark.yarn.submit.waitAppCompletion=true",
                "--py-files", py_files,
                script_s3,
            ] + args,
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--crawl", required=True)
    p.add_argument("--key-name", required=True)
    p.add_argument("--subnet-id", default="")
    p.add_argument("--skip-upload", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bucket = args.bucket
    crawl  = args.crawl
    scripts_prefix = "cc-political/scripts"

    s3 = boto3.client("s3", region_name=REGION)
    emr = boto3.client("emr", region_name=REGION)

    if not args.skip_upload:
        print("Uploading pipeline scripts to S3...")
        upload_scripts(s3, bucket, ".", scripts_prefix)

    def script(name):
        return f"s3://{bucket}/{scripts_prefix}/{name}"

    def s3out(stage):
        return f"s3://{bucket}/cc-political/{stage}/{crawl}/"

    py_files = script("config.py")

    steps = [
        make_spark_step(
            "1-FilterCCIndex",
            script("ingestion/crawl_filter.py"),
            ["--crawl", crawl,
             "--output", s3out("segments")],
            py_files,
        ),
        make_spark_step(
            "2-DownloadWARCs",
            script("ingestion/download_warcs.py"),
            ["--manifest", s3out("segments"),
             "--output",   s3out("raw-html")],
            py_files,
        ),
        make_spark_step(
            "3-ExtractText",
            script("processing/extract_text.py"),
            ["--input",  s3out("raw-html"),
             "--output", s3out("clean-text"),
             "--crawl",  crawl],
            py_files,
        ),
        make_spark_step(
            "4-CleanText",
            script("processing/clean_text.py"),
            ["--input",  s3out("clean-text"),
             "--output", s3out("corpus")],
            py_files,
        ),
        make_spark_step(
            "5-Sentiment",
            script("analysis/sentiment_analysis.py"),
            ["--input",  s3out("corpus"),
             "--output", s3out("sentiment"),
             "--aggregate"],
            py_files,
        ),
        make_spark_step(
            "6-TopicModeling",
            script("analysis/topic_modeling.py"),
            ["--input",       s3out("corpus"),
             "--output",      s3out("topics"),
             "--num-topics",  "30",
             "--max-iter",    "20"],
            py_files,
        ),
        make_spark_step(
            "7-IdeologyScoring",
            script("analysis/ideology_scoring.py"),
            ["--sentiment",     s3out("sentiment"),
             "--topics",        s3out("topics"),
             "--output",        s3out("ideology"),
             "--topic-labels",  f"s3://{bucket}/cc-political/config/topic_labels.json"],
            py_files,
        ),
        make_spark_step(
            "8-Clustering",
            script("clustering/polarization_clusters.py"),
            ["--input",  s3out("ideology") + "domain_ideology/",
             "--output", s3out("clusters"),
             "--k", "5"],
            py_files,
        ),
    ]

    network_cfg = {}
    if args.subnet_id:
        network_cfg["SubnetId"] = args.subnet_id

    response = emr.run_job_flow(
        Name=f"PoliticalPolarization-{crawl}",
        ReleaseLabel=EMR_RELEASE,
        Applications=[{"Name": "Spark"}, {"Name": "Hadoop"}],
        Instances={
            "MasterInstanceType": MASTER_INSTANCE,
            "SlaveInstanceType":  CORE_INSTANCE,
            "InstanceCount":      CORE_COUNT + 1,
            "Ec2KeyName":         args.key_name,
            "KeepJobFlowAliveWhenNoSteps": False,
            **network_cfg,
        },
        BootstrapActions=[
            {
                "Name": "InstallDependencies",
                "ScriptBootstrapAction": {
                    "Path": script("emr/bootstrap.sh"),
                },
            }
        ],
        Steps=steps,
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        LogUri=f"s3://{bucket}/cc-political/emr-logs/",
        VisibleToAllUsers=True,
        Tags=[
            {"Key": "Project", "Value": "PoliticalPolarization"},
            {"Key": "Crawl",   "Value": crawl},
        ],
    )

    cluster_id = response["JobFlowId"]
    print(f"\nEMR cluster launched: {cluster_id}")
    print(f"   Monitor at: https://console.aws.amazon.com/emr/home?region={REGION}#/clusterDetails/{cluster_id}")
    print(f"   Steps: {len(steps)} pipeline stages submitted")
    print(f"   Output root: s3://{bucket}/cc-political/")


if __name__ == "__main__":
    main()
