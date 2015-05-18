import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Favio on 15/05/15.
 */

object DataTypesMLlib {
def main(args: Array[String]) {

  val conf = new SparkConf()
//  .setMaster("local")
    .setMaster("mesos://master.mcbo.mood.com.ve:5050")
    .setAppName("Data Types MLlib")
    .set("spark.executor.memory", "6g")
  val sc = new SparkContext(conf)

//  /** MLlib supports local vectors and matrices stored on a single machine,
//    * as well as distributed matrices backed by one or more RDDs. Local vectors
//    * and local matrices are simple data models that serve as public interfaces.
//    * The underlying linear algebra operations are provided by Breeze and jblas.
//    */
//
//  //  1. Local Vector
//
//  /** A local vector has integer-typed and 0-based indices and double-typed
//    * values, stored on a single machine. MLlib supports two types of local
//    * vectors: dense and sparse. A dense vector is backed by a double array
//    * representing its entry values, while a sparse vector is backed by two
//    * parallel arrays: indices and values. For example, a vector (1.0, 0.0, 3.0)
//    * can be represented in dense format as [1.0, 0.0, 3.0] or in sparse format
//    * as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector.
//    */
//
//  /**
//   * The base class of local vectors is Vector, and we provide two
//   * implementations: DenseVector and SparseVector. We recommend using
//   * the factory methods implemented in Vectors to create local vectors.
//   */
//
//  // DenseVector(values: Array[Double])
//
//  // Create a dense vector (1.0, 0.0, 3.0).
//  val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
//  println(dv)
//
//  //  SparseVector(size: Int, indices: Array[Int], values: Array[Double])
//
//  // Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
//  val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
//  println(sv1)
//  // Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries.
//  val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))
//  println(sv2)
//
//
//
//  //  2. Labeled Point
//
//  /**
//   * A labeled point is a local vector, either dense or sparse, associated
//   * with a label/response. In MLlib, labeled points are used in supervised
//   * learning algorithms. We use a double to store a label, so we can use
//   * labeled points in both regression and classification. For binary
//   * classification, a label should be either 0 (negative) or 1 (positive).
//   * For multiclass classification, labels should be class indices starting
//   * from zero: 0, 1, 2, ....
//   */
//
//  //  LabeledPoint(label: Double, features: Vector)
//
//  // Create a labeled point with a positive label and a dense feature vector.
//  val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
//  println(pos)
//
//  // Create a labeled point with a negative label and a sparse feature vector.
//  val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
//  println(neg)
//
//  /**
//   * Sparse data
//   *
//   * It is very common in practice to have sparse training data. MLlib
//   * supports reading training examples stored in LIBSVM format, which
//   * is the default format used by LIBSVM and LIBLINEAR. It is a text
//   * format in which each line represents a labeled sparse feature vector
//   * using the following format:
//   *
//   * label index1:value1 index2:value2 ...
//   *
//   * where the indices are one-based and in ascending order. After loading,
//   * the feature indices are converted to zero-based.
//   */
//
//  //  MLUtils.loadLibSVMFile reads training examples stored in LIBSVM format.
//
//  val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/opt/spark/data/mllib/sample_libsvm_data.txt")
//  examples.foreach(println)
//  //  3. Local Matrix
//
//  /**
//   * A local matrix has integer-typed row and column indices and double-typed
//   * values, stored on a single machine. MLlib supports dense matrices, whose
//   * entry values are stored in a single double array in column major.
//   */
//
//  //  DenseMatrix(numRows: Int, numCols: Int, values: Array[Double])
//
//  // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
//  val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
//  println(dm)
//
//  //  4. Distributed Matrix
//
//  /**
//   * A distributed matrix has long-typed row and column indices and
//   * double-typed values, stored distributively in one or more RDDs.
//   * It is very important to choose the right format to store large
//   * and distributed matrices. Converting a distributed matrix to a
//   * different format may require a global shuffle, which is quite
//   * expensive. Three types of distributed matrices have been implemented
//   * so far.
//   */
//
//  /**
//   * The basic type is called RowMatrix. A RowMatrix is a row-oriented
//   * distributed matrix without meaningful row indices, e.g., a collection
//   * of feature vectors. It is backed by an RDD of its rows, where each
//   * row is a local vector. We assume that the number of columns is not
//   * huge for a RowMatrix so that a single local vector can be reasonably
//   * communicated to the driver and can also be stored / operated on using
//   * a single node. An IndexedRowMatrix is similar to a RowMatrix but with
//   * row indices, which can be used for identifying rows and executing
//   * joins. A CoordinateMatrix is a distributed matrix stored in coordinate
//   * list (COO) format, backed by an RDD of its entries.
//   */
//
//  /**
//   * COO stores a list of (row, column, value) tuples. Ideally,
//   * the entries are sorted (by row index, then column index) to
//   * improve random access times. This is another format which is good
//   * for incremental matrix construction.
//   */
//
//  /**
//   * Note: The underlying RDDs of a distributed matrix must be
//   * deterministic, because we cache the matrix size. In general the
//   * use of non-deterministic RDDs can lead to errors.
//   */
//
//  //  5. Block Matrix
//
//  /**
//   * A BlockMatrix is a distributed matrix backed by an RDD of
//   * MatrixBlocks, where a MatrixBlock is a tuple of ((Int, Int), Matrix),
//   * where the (Int, Int) is the index of the block, and Matrix is the
//   * sub-matrix at the given index with size rowsPerBlock x colsPerBlock.
//   * BlockMatrix supports methods such as add and multiply with another
//   * BlockMatrix. BlockMatrix also has a helper function validate which
//   * can be used to check whether the BlockMatrix is set up properly.
//   */
//
//  /**
//   * A BlockMatrix can be most easily created from an IndexedRowMatrix or
//   * CoordinateMatrix by calling toBlockMatrix. toBlockMatrix creates
//   * blocks of size 1024 x 1024 by default. Users may change the block
//   * size by supplying the values through toBlockMatrix(rowsPerBlock,
//   * colsPerBlock).
//   */

//  val entries: RDD[MatrixEntry] = sc.parallelize(Seq(
//    (0, 0, 1.0),
//    (0, 1, 2.0),
//    (1, 1, 3.0),
//    (1, 2, 4.0),
//    (2, 2, 5.0),
//    (2, 3, 6.0),
//    (3, 0, 7.0),
//    (3, 3, 8.0),
//    (4, 1, 9.0))
//    .map { case (i, j, value) => MatrixEntry(i, j, value) })
//
//  // Create a CoordinateMatrix from an RDD[MatrixEntry].
//  val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)
//
//  // Transform the CoordinateMatrix to a BlockMatrix
//  val matA: BlockMatrix = coordMat.toBlockMatrix().cache()
//
//  // Validate whether the BlockMatrix is set up properly. Throws an Exception when it is not valid.
//  // Nothing happens if it is valid.
//  matA.validate()
//
//  // Calculate A^T * A.
//  val ata = matA.transpose.multiply(matA)
//  println(ata.toLocalMatrix())

//  6. RowMatrix

//  /**
//   * A RowMatrix is a row-oriented distributed matrix without meaningful
//   * row indices, backed by an RDD of its rows, where each row is a local
//   * vector. Since each row is represented by a local vector, the number of
//   * columns is limited by the integer range but it should be much smaller
//   * in practice.
//   */
//
//  /**
//   * A RowMatrix can be created from an RDD[Vector] instance. Then we can
//   * compute its column summary statistics.
//   */
//
//  val rows: RDD[Vector] = sc.parallelize(Seq(0.0,1.0,2.0)
//  .map {i => Vectors.dense(i) } )
//
//  // Create a RowMatrix from an RDD[Vector].
//  val mat: RowMatrix = new RowMatrix(rows)
//
//  // Get its size.
//  val m = mat.numRows()
//  val n = mat.numCols()
//  println(m,n)

//  7. IndexedRowMatrix

  /**
   * An IndexedRowMatrix is similar to a RowMatrix but with meaningful
   * row indices. It is backed by an RDD of indexed rows, so that each
   * row is represented by its index (long-typed) and a local vector.
   */

  /**
   * An IndexedRowMatrix can be created from an RDD[IndexedRow] instance,
   * where IndexedRow is a wrapper over (Long, Vector). An IndexedRowMatrix
   * can be converted to a RowMatrix by dropping its row indices.
   */
   
//TODO: Finish section
  val rows1: RDD[IndexedRow] = sc.parallelize(Seq((1,Vectors.dense(0.0,1.0,2.0)))
    .map {case (a,b) => IndexedRow(a,b)})

  val mat1: IndexedRowMatrix = new IndexedRowMatrix(rows1)

  //Gets its size
  val m1 = mat1.numRows()
  val n1 = mat1.numCols()

  //Drop its row indices
  val rowMat: RowMatrix = mat1.toRowMatrix()

  sc.stop()
  }
}
