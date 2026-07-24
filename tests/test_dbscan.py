from pyspark.sql import types as t
from sonardbscan import SonarDBSCAN


def test_sonar_dbscan_clustering(spark):
    """
    Validates the end-to-end clustering pipeline of the SonarDBSCAN model.

    Evaluates the algorithm's ability to correctly assign cluster IDs to dense
    regions, effectively separate distinct clusters, and accurately flag isolated
    points as noise (-1).

    Args:
        spark (SparkSession): The active test Spark session.
    """
    data = [
        (1, [0.0, 0.0]), (2, [0.0, 0.1]), (3, [0.0, 0.2]),
        (4, [10.0, 10.0]), (5, [10.1, 10.1]), (6, [9.9, 10.0]),
        (7, [5.0, 5.0]), (8, [5.1, 4.9])
    ]

    schema = t.StructType([
        t.StructField('identifier', t.IntegerType(), nullable=False),
        t.StructField('features', t.ArrayType(t.DoubleType()), nullable=False)
    ])

    df = spark.createDataFrame(data, schema)

    dbscan = SonarDBSCAN(epsilon=0.3, min_pts=2, features_col='features', identifier_col='identifier')
    result_df = dbscan.fit(df)

    results = result_df.collect()
    cluster_mapping = {row['identifier']: row['clusterId'] for row in results}

    assert cluster_mapping[1] == cluster_mapping[2] == cluster_mapping[3]
    assert cluster_mapping[4] == cluster_mapping[5] == cluster_mapping[6]
    assert cluster_mapping[1] != cluster_mapping[4]

    assert cluster_mapping[7] == -1
    assert cluster_mapping[8] == -1
