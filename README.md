# SonarDBSCAN 

SonarDBSCAN is a highly optimized, distributed implementation of the **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** algorithm, built natively for PySpark. 

By leveraging **GraphFrames** for connected component resolution and a **Triangle Inequality-based metric space optimization**, SonarDBSCAN drastically reduces the computational overhead typically associated with distributed spatial clustering.

---

## 📖 The Algorithm

Standard DBSCAN relies on two hyper-parameters:
* **Epsilon (eps):** The spatial radius around a given point.
* **MinPts (n):** The minimum number of neighbors within the `eps` radius required to define a dense region.

The algorithm categorizes points into three types:
1. **Core Points:** Points with at least `MinPts` neighbors.
2. **Border Points:** Points with fewer than `MinPts` neighbors, but which fall within the `eps` radius of a Core Point.
3. **Noise:** Points that are neither Core nor Border points.

### The PySpark Challenge & Our Optimization
In a distributed environment like PySpark, calculating distances between all pairs of points requires a massive Cartesian product (O(N²)), which causes severe network shuffling and memory bottlenecks.

SonarDBSCAN solves this in metric spaces (like Euclidean distance) using the **Triangle Inequality**: 
`d(a, b) + d(b, c) >= d(a, c)`

**The Ring Partitioning Strategy:**
1. The algorithm selects a random pivot point, `c`.
2. It calculates the distance from all dataset points to `c`.
3. It assigns each point to a "ring" `k`, such that its distance to `c` falls exactly between `k * eps` and `(k + 1) * eps`.

Based on this projection, we can mathematically eliminate the need to compare points across distant rings using two geometric lemmas:

* **Lemma 1:** If `d(x, c) >= (k + 1) * eps` and `d(y, c) < k * eps`, then `d(x, y) > eps`.
  *(Proof: `d(x, y) >= d(x, c) - d(y, c) > (k + 1)*eps - k*eps = eps`)*
* **Lemma 2:** If `d(x, c) <= k * eps` and `d(y, c) > (k + 1) * eps`, then `d(x, y) > eps`.
  *(Proof: `d(x, c) >= d(y, c) - d(x, y) > (1 + k)*eps - k*eps = eps`)*

**The Result:** If point `x` is in ring `k`, SonarDBSCAN only evaluates distances between `x` and points in rings `k-1`, `k`, and `k+1`. All other Cartesian pairs are safely discarded prior to any shuffling, resulting in massive performance gains.

---

## ⚙️ Prerequisites & Installation

SonarDBSCAN requires PySpark and the **GraphFrames** package. We recommend using `uv` for fast, reproducible environment management.

### 1. Fetch the GraphFrames JAR
GraphFrames is required for the underlying graph component processing. You must download the JAR file that matches your Spark and Scala versions. 

For **Spark 3.2 (Scala 2.12)**, open your terminal and fetch it using `curl`:

```bash
curl -O https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar
```

### 2. Install SonarDBSCAN
From the root of this project directory, create a virtual environment and install the library using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install .
```

---

## 🚀 Usage Guide

SonarDBSCAN is designed to be plug-and-play for PySpark DataFrames. 

### Step 1: Initialize the Spark Session
Because GraphFrames relies on connected components (which recursively generate long RDD lineages), it **requires** a Spark Checkpoint directory to truncate the lineage and prevent StackOverflow errors. 

You must attach the JAR and set the checkpoint directory when starting your session:

```python
from pyspark.sql import SparkSession

# Path to the JAR you downloaded via curl
GRAPH_FRAMES_JAR_PATH = "graphframes-0.8.2-spark3.2-s_2.12.jar"

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SonarDBSCAN_Clustering") \
    .config('spark.jars', GRAPH_FRAMES_JAR_PATH) \
    .getOrCreate()

# REQUIRED: Set a checkpoint directory for GraphFrames
spark.sparkContext.setCheckpointDir('/tmp/spark-checkpoints')
```

### Step 2: Prepare Your Data
Your DataFrame must have two specific columns:
1. An **identifier** column (`IntegerType`)
2. A **features** column (`ArrayType(DoubleType)`)

```python
from pyspark.sql import types as t

data = [
    (1, [0.0, 0.0]), (2, [0.0, 0.1]), (3, [0.0, 0.2]), # Will form Cluster 1
    (4, [10.0, 10.0]), (5, [10.1, 10.1]),              # Will form Cluster 2
    (7, [5.0, 5.0]), (8, [5.1, 4.9])                   # Isolated points (Noise)
]

schema = t.StructType([
    t.StructField('identifier', t.IntegerType(), nullable=False),
    t.StructField('features', t.ArrayType(t.DoubleType()), nullable=False)
])

df = spark.createDataFrame(data, schema)
```

### Step 3: Run the Clustering
Instantiate the `SonarDBSCAN` class and call the `.fit()` method. 

```python
from sonardbscan import SonarDBSCAN

# Initialize the model
scanner = SonarDBSCAN(
    epsilon=0.3, 
    min_pts=2, 
    features_col='features', 
    identifier_col='identifier'
)

# Fit the model and generate clusters
clustered_df = scanner.fit(df)

# View the results
clustered_df.select('identifier', 'features', 'clusterId').show()
```

*Note: The output DataFrame includes a new `clusterId` column. Any point classified as Noise is assigned a `clusterId` of `-1`.*

---

## 🧪 Running Tests

SonarDBSCAN includes a comprehensive Pytest suite. The test configuration automatically downloads the GraphFrames dependency dynamically, so you don't need to manually link the JAR for testing.

To run the tests with `uv`:

```bash
# Install the testing dependencies
uv pip install -e .[dev]

# Run the test suite
pytest tests/
```

---

## 🗺️ Roadmap & Known Limitations
* **Non-Sequential Cluster IDs:** Because cluster IDs are generated via GraphFrames' connected components (using the lowest vertex ID in the component), the output `clusterId` integers may skip numbers (e.g., clusters 1, 4, 7). This does not affect mathematical correctness but is purely aesthetic.
* **Algorithm Extensibility:** Future versions plan to support configurable distance metrics (e.g., Manhattan, Cosine) provided they satisfy the metric space rules necessary for the triangle inequality optimization.