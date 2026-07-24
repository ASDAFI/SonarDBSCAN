from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f, Window
from graphframes import GraphFrame

from .distances import find_epsilon_pairs


class SonarDBSCAN:
    """
    Distributed Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    This class provides a scalable implementation of the DBSCAN algorithm for PySpark
    DataFrames. It utilizes GraphFrames for efficient connected-component resolution
    and implements a metric-space optimization via triangle inequality to heavily reduce
    the computational overhead of distance calculations.

    Attributes:
        epsilon (float): The spatial radius threshold for neighborhood queries.
        min_pts (int): The minimum number of points required to form a dense region.
        features_col (str): The column name representing the feature vectors.
        identifier_col (str): The column name representing the unique point identifiers.
    """

    def __init__(self, epsilon: float, min_pts: int, features_col: str = 'features',
                 identifier_col: str = 'identifier'):
        """
        Initializes the SonarDBSCAN clustering model.

        Args:
            epsilon (float): The maximum distance between two points for one to be
                considered in the neighborhood of the other.
            min_pts (int): The minimum number of neighbors required to designate a core point.
            features_col (str, optional): The name of the features column. Defaults to 'features'.
            identifier_col (str, optional): The name of the identifier column. Defaults to 'identifier'.
        """
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.features_col = features_col
        self.identifier_col = identifier_col

    def fit(self, df: DataFrame) -> DataFrame:
        """
        Executes the DBSCAN clustering algorithm on the provided dataset.

        Args:
            df (DataFrame): The input PySpark DataFrame containing the data points
                to be clustered. Must contain the identifier and features columns.

        Returns:
            DataFrame: A PySpark DataFrame containing the original data appended
            with a 'clusterId' column. Noise points are assigned a cluster ID of -1.
        """
        paired_df = find_epsilon_pairs(df, self.features_col, self.identifier_col, self.epsilon)

        centers_df = paired_df.groupBy(f'point1.{self.identifier_col}', f'point1.{self.features_col}') \
            .agg(f.count(f.lit(1)).alias('countNeighbors')) \
            .filter(f.col('countNeighbors') >= self.min_pts)

        edges_df = find_epsilon_pairs(centers_df, self.features_col, self.identifier_col, self.epsilon) \
            .select(f.col(f'point1.{self.identifier_col}').alias('src'),
                    f.col(f'point2.{self.identifier_col}').alias('dst'))

        nodes_df = centers_df.select(f.col(self.identifier_col).alias('id'))

        graph = GraphFrame(nodes_df, edges_df)
        connected_components_centers_df = graph.connectedComponents() \
            .select(f.col('id').alias(self.identifier_col), f.col('component').alias('clusterId'))

        cluster_id_df = df.join(connected_components_centers_df, on=self.identifier_col, how='left')

        cluster_id_paired_df = find_epsilon_pairs(cluster_id_df, self.features_col, self.identifier_col, self.epsilon) \
            .filter((f.col('point1.clusterId').isNull()) & (f.col('point2.clusterId').isNotNull()))

        window_spec = Window.partitionBy(f'point1.{self.identifier_col}', f'point1.{self.features_col}').orderBy(
            "distance")

        non_noise_clustered_df = cluster_id_paired_df.withColumn("row_num", f.row_number().over(window_spec)) \
            .filter(f.col('row_num') == f.lit(1)) \
            .select(f'point1.{self.identifier_col}', f'point1.{self.features_col}', 'point2.clusterId')

        clustering_output_df = df.join(connected_components_centers_df, on=self.identifier_col, how='left') \
            .join(non_noise_clustered_df.withColumnRenamed('clusterId', 'clusterId2').drop(self.features_col),
                  on=self.identifier_col, how='left') \
            .withColumn('clusterId',
                        f.when(f.col('clusterId').isNotNull(), f.col('clusterId')).otherwise(f.col('clusterId2'))) \
            .drop('clusterId2') \
            .fillna({'clusterId': -1})

        return clustering_output_df