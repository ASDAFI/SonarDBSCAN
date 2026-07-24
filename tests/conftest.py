import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """
    Initializes and yields a configured SparkSession for the testing environment.

    This fixture dynamically downloads and attaches the required GraphFrames
    package and configures a temporary checkpoint directory required for
    evaluating connected components.

    Yields:
        SparkSession: The active Spark session for executing PySpark tests.
    """
    spark_session = SparkSession.builder \
        .master("local[*]") \
        .appName("SonarDBSCAN_TestSuite") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
        .getOrCreate()

    spark_session.sparkContext.setCheckpointDir('/tmp/checkpoints')

    yield spark_session

    spark_session.stop()
