import breeze.linalg.DenseVector
import com.sircamp.utils.Metrics
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers, PrivateMethodTester}

@RunWith(classOf[JUnitRunner])
class MetricsSuite extends FunSuite with BeforeAndAfterAll with Matchers with PrivateMethodTester{

  var yPred:DenseVector[Int] = _
  var yTrue:DenseVector[Int] = _

  var yPredExcept:DenseVector[Int] = _
  var yTrueExcept:DenseVector[Int] = _


  override protected def beforeAll(): Unit = {
    super.beforeAll()

    val yTrueArray = Array(0, 1, 1, 0, 2, 1, 2, 1, 0, 0, 2)
    val yPredArray = Array(0, 1, 2, 0, 0, 1, 2, 1, 1, 0, 2)

    yPred = new DenseVector[Int](yPredArray)
    yTrue = new DenseVector[Int](yTrueArray)

    val yTrueExceptArray = Array(0, 1, 1, 0, 2, 1, 2, 1, 0, 0, 2)
    val yPredExceptArray = Array(0, 1, 2, 0, 0, 1, 2, 1, 1, 0 )

    yPredExcept = new DenseVector[Int](yPredExceptArray)
    yTrueExcept = new DenseVector[Int](yTrueExceptArray)
  }

  test("Test accuracyScore"){

    val accuracy = Metrics.accuracyScore(yTrue, yPred)
    println("Accuracy: "+accuracy)

    assert(accuracy == 0.7272727272727273)

  }

  test("Test precisionScore"){

    val precision = Metrics.precisionScore(yTrue, yPred)
    println("Precision: "+precision)

    assert(precision == 0.8571428571428571)

  }


  test("Test recallScore"){

    val recall = Metrics.recallScore(yTrue, yPred)
    println("Recall: "+recall)

    assert(recall == 0.8571428571428571)

  }


  test("Test F1Score"){

    val f1 = Metrics.f1Score(yTrue, yPred)
    println("F1: "+f1)

    assert(f1 == 0.8571428571428571)

  }

  test("Test computeConfusionMatrixValues"){

    val computeConfusionMatrixValues = PrivateMethod[(Int,Int,Int,Int)]('computeConfusionMatrixValues)
    val result:(Int,Int,Int,Int) = Metrics invokePrivate  computeConfusionMatrixValues(yTrue, yPred)

    assert((6,3,1,1) == result)
  }

  test("Test checkDims Exception"){

    val checkDims = PrivateMethod[Unit]('checkDims)
    an [IllegalArgumentException] should be thrownBy { Metrics invokePrivate  checkDims(yTrueExcept, yPredExcept) }
  }

}
