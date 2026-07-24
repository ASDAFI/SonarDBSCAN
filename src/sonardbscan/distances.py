import math

from pyspark.sql import functions as f, types as t
from pyspark.sql import DataFrame


@f.udf(returnType=t.DoubleType())
def euclidean_distance(v1: list, v2: list) -> float:
    """
    Calculates the Euclidean distance between two n-dimensional vectors.

    Args:
        v1 (list): The first vector represented as a list of numerical values.
        v2 (list): The second vector represented as a list of numerical values.

    Returns:
        float: The computed Euclidean distance between the two input vectors.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def find_epsilon_pairs(df: DataFrame, features_col: str, identifier_col: str, eps: float) -> DataFrame:
    """
    Identifies pairs of points within a specified epsilon distance.

    This function optimizes the distributed distance computation by employing
    a ring-partitioning strategy based on the triangle inequality. By selecting
    a random pivot point and assigning spatial rings, it minimizes the number
    of required Cartesian joins and distance evaluations across the dataset.

    Args:
        df (DataFrame): The input PySpark DataFrame containing the data points.
        features_col (str): The name of the column containing the feature vectors.
        identifier_col (str): The name of the column containing unique point identifiers.
        eps (float): The maximum spatial radius (epsilon) to consider two points as neighbors.

    Returns:
        DataFrame: A PySpark DataFrame containing valid pairs of points that
        are within the epsilon distance, along with their computed distance.
    """
    random_value = df.select(features_col).rdd.takeSample(False, 1)[0][0]

    df_ringed = df.withColumn('sampleCenter', f.lit(random_value)) \
        .withColumn('ringId',
                    (euclidean_distance(f.col(features_col), f.col('sampleCenter')) / f.lit(eps)).cast(t.IntegerType())) \
        .drop('sampleCenter') \
        .orderBy('ringId')

    paired_df = df_ringed.alias('point1').join(
        df_ringed.alias('point2'),
        on=(f.abs(f.col('point1.ringId') - f.col('point2.ringId')) <= 1)
    ).filter(
        f.col(f'point1.{identifier_col}') != f.col(f'point2.{identifier_col}')
    ).drop('point1.ringId', 'point2.ringId')

    paired_df = paired_df.withColumn(
        'distance',
        euclidean_distance(f.col(f'point1.{features_col}'), f.col(f'point2.{features_col}'))
    ).filter(f.col('distance') <= f.lit(eps))

    return paired_df