
# SonarDBSCAN

SonarDBSCAN is a highly optimized, distributed implementation of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm, built natively for PySpark. 

Designed for large-scale spatial datasets, SonarDBSCAN mitigates the $O(N^2)$ computational bottleneck of distributed distance calculations by leveraging a metric-space optimization based on the triangle inequality. Graph processing and connected component resolution are delegated to the GraphFrames library, ensuring robust lineage handling and execution stability across cluster environments.

---

## Table of Contents
1. [Mathematical Foundations and Optimization](#mathematical-foundations-and-optimization)
2. [System Requirements and Installation](#system-requirements-and-installation)
3. [Usage Integration](#usage-integration)
4. [Testing Framework](#testing-framework)
5. [Performance Considerations and Limitations](#performance-considerations-and-limitations)

---

## Mathematical Foundations and Optimization

Standard DBSCAN implementations in distributed environments suffer from the necessity of calculating the Cartesian product of all data points to discover $\epsilon$-neighborhoods. This results in severe network shuffling and memory exhaustion.

SonarDBSCAN assumes the underlying space is a metric space $(X, d)$, satisfying the triangle inequality:

$$d(x, y) + d(y, c) \geq d(x, c)$$

To optimize neighborhood discovery, the algorithm implements a Ring Partitioning strategy:
1. A global pivot point $c \in X$ is selected uniformly at random.
2. The distance $d(x, c)$ is computed for every point $x \in X$.
3. Points are partitioned into disjoint spatial rings $R_k$, where a point $x$ belongs to ring $k$ if and only if its distance to $c$ satisfies $k\epsilon \leq d(x, c) < (k + 1)\epsilon$.

By projecting the dataset into these rings, SonarDBSCAN mathematically guarantees that Cartesian evaluations are only necessary between adjacent rings. This guarantee is formalized by the following lemmas:

**Lemma 1**
If $d(x, c) \geq (k + 1)\epsilon$ and $d(y, c) < k\epsilon$, then $d(x, y) > \epsilon$.

*Proof:*
By the triangle inequality, we have $d(x, y) + d(y, c) \geq d(x, c)$. Rearranging the terms yields:
$$d(x, y) \geq d(x, c) - d(y, c)$$
Substituting the known bounds:
$$d(x, y) > (k + 1)\epsilon - k\epsilon = \epsilon$$

**Lemma 2**
If $d(x, c) \leq k\epsilon$ and $d(y, c) > (k + 1)\epsilon$, then $d(x, y) > \epsilon$.

*Proof:*
By the triangle inequality, $d(x, y) + d(x, c) \geq d(y, c)$. Rearranging the terms yields:
$$d(x, y) \geq d(y, c) - d(x, c)$$
Substituting the known bounds:
$$d(x, y) > (1 + k)\epsilon - k\epsilon = \epsilon$$

**Consequence:** For any point $x \in R_k$, neighbors within distance $\epsilon$ can strictly only exist within $R_{k-1}$, $R_k$, and $R_{k+1}$. All other Cartesian pairs are pruned prior to the Spark shuffle phase.

## System Requirements and Installation

SonarDBSCAN requires PySpark ($\geq 3.0.0$) and GraphFrames ($\geq 0.8.0$). It is recommended to manage the environment using `uv` for deterministic dependency resolution.

### 1. Retrieve the GraphFrames Dependency
GraphFrames relies on a compiled JAR that must match your Spark and Scala versions. For Spark 3.2 built on Scala 2.12, fetch the JAR via your terminal:

```bash
curl -O https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar
```

### 2. Package Installation
Instantiate a virtual environment and install the package from the source directory:

```bash
uv venv
source .venv/bin/activate
uv pip install .
```

## Usage Integration

The library interfaces directly with PySpark DataFrames. The input DataFrame strictly requires an integer identifier column and a features array column of double-precision floats.

### Session Initialization
GraphFrames' connected component algorithm builds deep RDD lineages. To prevent memory overflow and `StackOverflowError` exceptions, a Spark Checkpoint directory must be explicitly configured during session initialization.

```python
from pyspark.sql import SparkSession

GRAPH_FRAMES_JAR_PATH = "graphframes-0.8.2-spark3.2-s_2.12.jar"

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SonarDBSCAN_Execution") \
    .config("spark.jars", GRAPH_FRAMES_JAR_PATH) \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")
```

### Execution Example

```python
from pyspark.sql import types as t
from sonardbscan import SonarDBSCAN

data = [
    (1, [0.0, 0.0]), (2, [0.0, 0.1]), (3, [0.0, 0.2]),
    (4, [10.0, 10.0]), (5, [10.1, 10.1]),
    (7, [5.0, 5.0]), (8, [5.1, 4.9])
]

schema = t.StructType([
    t.StructField("identifier", t.IntegerType(), nullable=False),
    t.StructField("features", t.ArrayType(t.DoubleType()), nullable=False)
])

df = spark.createDataFrame(data, schema)

# Initialize the clustering model
scanner = SonarDBSCAN(
    epsilon=0.3, 
    min_pts=2, 
    features_col="features", 
    identifier_col="identifier"
)

# Execute clustering
clustered_df = scanner.fit(df)
clustered_df.select("identifier", "features", "clusterId").show()
```

*Implementation Note: Observations categorized as Noise by the DBSCAN definition are deterministically assigned a `clusterId` of `-1`.*

## Testing Framework

The repository is equipped with a comprehensive Pytest suite covering distance metrics, triangle inequality boundaries, and complete graph resolution. The test configuration automatically dynamically resolves the GraphFrames dependencies.

```bash
uv pip install -e .[dev]
pytest tests/
```

## Performance Considerations and Limitations

* **Discontinuous Cluster Identifiers:** Because cluster indices are derived directly from the lowest vertex identifier within a GraphFrames connected component, the resulting `clusterId` values are not guaranteed to be strictly sequential (e.g., outputs may yield indices 1, 4, and 7 without intermediate values).
* **Metric Space Constraint:** The internal optimization is fundamentally reliant on the triangle inequality. Custom distance functions injected into the architecture must mathematically satisfy metric space axioms; otherwise, the spatial pruning will yield false negatives during neighbor detection.

