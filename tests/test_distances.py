from pyspark.sql import types as t
from sonardbscan.distances import find_epsilon_pairs


def test_find_epsilon_pairs(spark):
    """
    Verifies the correctness of the triangle inequality ring-partitioning logic.

    Ensures that the algorithm accurately identifies spatial pairs within the
    specified epsilon distance while correctly filtering out points that exceed
    the threshold.

    Args:
        spark (SparkSession): The active test Spark session.
    """
    data = [
        (1, [0.0, 0.0]),
        (2, [0.1, 0.0]),
        (3, [10.0, 10.0])
    ]
    schema = t.StructType([
        t.StructField('identifier', t.IntegerType(), nullable=False),
        t.StructField('features', t.ArrayType(t.DoubleType()), nullable=False)
    ])
    df = spark.createDataFrame(data, schema)

    paired_df = find_epsilon_pairs(df, 'features', 'identifier', 0.5)
    results = paired_df.collect()

    assert len(results) == 2

    id_pairs = set((row['point1.identifier'], row['point2.identifier']) for row in results)
    assert (1, 2) in id_pairs
    assert (2, 1) in id_pairs
    assert (1, 3) not in id_pairs
