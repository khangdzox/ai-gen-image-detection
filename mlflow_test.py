import os
import mlflow

os.environ["AWS_ACCESS_KEY_ID"] = "khangvo3103"
os.environ["AWS_SECRET_ACCESS_KEY"] = "vk3103@minio"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://103.21.1.103:25001"

mlflow.set_tracking_uri("http://103.21.1.103:25000")
mlflow.set_experiment("test_experiment")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("param1", "value1")
    mlflow.log_param("param2", "value2")

    mlflow.log_metric("metric1", 0.1)
    mlflow.log_metric("metric2", 0.2)

    with open("test.txt", "w") as f:
        f.write("test")

    mlflow.log_artifact("test.txt", artifact_path="test")
