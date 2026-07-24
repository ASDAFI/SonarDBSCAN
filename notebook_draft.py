import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as f, types as t, Window
import matplotlib.pyplot as plt

from graphframes import GraphFrame

GRAPH_FRAMES_JAR_PATH = "graphframes-0.8.2-spark3.2-s_2.12.jar"


spark = SparkSession.builder \
        .master("local[*]") \
        .appName("DBSCAN") \
        .config('spark.jars', GRAPH_FRAMES_JAR_PATH).getOrCreate()

spark.sparkContext.setCheckpointDir('/home/ali/Desktop/cassandra/dbscan/checkpoints')

############################################

data = [
    (1, [0.0, 0.0]),
    (2, [0.0, 0.1]),
    (3, [0.0, 0.2]),
    (4, [10.0, 10.0]),
    (5, [10.1, 10.1]),
    (6, [9.9, 10.0]),
    (7, [5.0, 5.0]),
    (8, [5.1, 4.9])
]


FEATURES_COL = 'features'
IDENTIFIERS_COL = 'indentifier'

schema = t.StructType([
    t.StructField(IDENTIFIERS_COL, t.IntegerType(), nullable=False),
    t.StructField(FEATURES_COL, t.ArrayType(t.DoubleType()), nullable=False)
])

df = spark.createDataFrame(data, schema)

############################################

EPSILON = 0.3
MINPTS = 2

############################################


@f.udf(returnType=t.DoubleType())
def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def find_vertices_pair_with_less_than_epsilon_distance(df, features_column, identifier_column, eps):

    random_value = df.select(features_column).rdd.takeSample(False, 1)[0][0]

    df = df.withColumn('sampleCenter', f.lit(random_value))\
            .withColumn('ringId',
                        (euclidean_distance(f.col(features_column), f.col('sampleCenter')) / f.lit(eps)).cast(t.IntegerType()))\
            .drop('sampleCenter')

    df = df.orderBy('ringId')
    paired_df = df.alias('point1').join(df.alias('point2'), on=(f.abs(f.col('point1.ringId') - f.col('point2.ringId')) <= 1))\
                                  .filter(f.col(f'point1.{identifier_column}') != f.col(f'point2.{identifier_column}'))\
                                  .drop('point1.ringId', 'point2.ringId')
    paired_df = paired_df.withColumn('distance', euclidean_distance(f.col(f'point1.{features_column}'), f.col(f'point2.{features_column}')))\
                         .filter(f.col('distance') <= f.lit(eps))
    return paired_df

paired_df = find_vertices_pair_with_less_than_epsilon_distance(df, FEATURES_COL, IDENTIFIERS_COL, EPSILON)

centers_df = paired_df.groupBy(f'point1.{IDENTIFIERS_COL}', f'point1.{FEATURES_COL}').agg(f.count(f.lit(1)).alias('countNeighbors'))\
                      .filter(f.col('countNeighbors') >= MINPTS)


edges_df = find_vertices_pair_with_less_than_epsilon_distance(centers_df, FEATURES_COL, IDENTIFIERS_COL, EPSILON)\
           .select(f.col(f'point1.{IDENTIFIERS_COL}').alias('src'), f.col(f'point2.{IDENTIFIERS_COL}').alias('dst'))
nodes_df = centers_df.select(f.col(IDENTIFIERS_COL).alias('id'))

graph = GraphFrame(nodes_df, edges_df)
connected_components_centers_df = graph.connectedComponents().select(f.col('id').alias(IDENTIFIERS_COL), f.col('component').alias('clusterId'))

cluster_id_df = df.join(connected_components_centers_df, on=IDENTIFIERS_COL, how='left')

# TODO: Optimize it by adding it to function
cluster_id_paired_df = find_vertices_pair_with_less_than_epsilon_distance(cluster_id_df, FEATURES_COL, IDENTIFIERS_COL, EPSILON)\
                        .filter((f.col('point1.clusterId').isNull()) & (f.col('point2.clusterId').isNotNull()))

# TODO: make it more efficient
window_spec = Window.partitionBy(f'point1.{IDENTIFIERS_COL}', f'point1.{FEATURES_COL}').orderBy("distance")

non_noise_clustered_df = cluster_id_paired_df.withColumn("row_num", f.row_number().over(window_spec))\
    .filter(f.col('row_num') == f.lit(1))\
    .select(f'point1.{IDENTIFIERS_COL}', f'point1.{FEATURES_COL}', 'point2.clusterId')

clustering_output_df = df.join(connected_components_centers_df, on=IDENTIFIERS_COL, how='left')\
                         .join(non_noise_clustered_df.withColumnRenamed('clusterId', 'clusterId2').drop(FEATURES_COL)
                                            , on=IDENTIFIERS_COL, how='left')\
                        .withColumn('clusterId', f.when(f.col('clusterId').isNotNull(), f.col('clusterId')).otherwise(f.col('clusterId2')))\
                        .drop('clusterId2').fillna({'clusterId': -1})

##########################


cluster_data = (
    clustering_output_df
    .select(IDENTIFIERS_COL, FEATURES_COL, "clusterId")
    .rdd
    .map(lambda row: (row[1][0], row[1][1], row[2]))
    .collect()
)

unique_clusters = sorted({point[2] for point in cluster_data})

plt.figure(figsize=(8, 6))

for cid in unique_clusters:
    x_points = [p[0] for p in cluster_data if p[2] == cid]
    y_points = [p[1] for p in cluster_data if p[2] == cid]

    if cid == -1:
        plt.scatter(x_points, y_points, marker='x', label='Noise')
    else:
        plt.scatter(x_points, y_points, label=f'Cluster {cid}')

plt.title('DBSCAN Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
