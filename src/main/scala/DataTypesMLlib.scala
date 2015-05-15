import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.log4j.Logger
import org.apache.log4j.Level

/**
 * Created by Favio on 15/05/15.
 */

object DataTypesMLlib {
def main(args: Array[String]) {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val conf = new SparkConf()
    //      .setMaster("local")
    .setMaster("mesos://master.mcbo.mood.com.ve:5050")
    .setAppName("Data Types MLlib")
  val sc = new SparkContext(conf)

  /** MLlib supports local vectors and matrices stored on a single machine,
  * as well as distributed matrices backed by one or more RDDs. Local vectors
  * and local matrices are simple data models that serve as public interfaces.
  * The underlying linear algebra operations are provided by Breeze and jblas.
  */

//  1. Local Vector

 /** A local vector has integer-typed and 0-based indices and double-typed
  * values, stored on a single machine. MLlib supports two types of local
  * vectors: dense and sparse. A dense vector is backed by a double array
  * representing its entry values, while a sparse vector is backed by two
  * parallel arrays: indices and values. For example, a vector (1.0, 0.0, 3.0)
  * can be represented in dense format as [1.0, 0.0, 3.0] or in sparse format
  * as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector.
  */

  /**
   * The base class of local vectors is Vector, and we provide two
   * implementations: DenseVector and SparseVector. We recommend using
   * the factory methods implemented in Vectors to create local vectors.
   */

// DenseVector(values: Array[Double])

  // Create a dense vector (1.0, 0.0, 3.0).
  val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
  println(dv)

//  SparseVector(size: Int, indices: Array[Int], values: Array[Double])

  // Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
  val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
  println(sv1)
  // Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries.
  val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))
  println(sv2)



//  2. Labeled Point

  /**
   * A labeled point is a local vector, either dense or sparse, associated
   * with a label/response. In MLlib, labeled points are used in supervised
   * learning algorithms. We use a double to store a label, so we can use
   * labeled points in both regression and classification. For binary
   * classification, a label should be either 0 (negative) or 1 (positive).
   * For multiclass classification, labels should be class indices starting
   * from zero: 0, 1, 2, ....
   */

//  LabeledPoint(label: Double, features: Vector)

  // Create a labeled point with a positive label and a dense feature vector.
  val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
  println(pos)

  // Create a labeled point with a negative label and a sparse feature vector.
  val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
  println(neg)

  sc.stop()
  }
}
