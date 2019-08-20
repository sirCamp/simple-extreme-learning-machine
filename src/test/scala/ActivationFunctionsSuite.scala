import breeze.linalg.{DenseMatrix, DenseVector}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import org.scalatest.junit.JUnitRunner

import scala.collection.mutable.ListBuffer

@RunWith(classOf[JUnitRunner])
class ActivationFunctionsSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  var X : DenseMatrix[Double] = _

  override protected def beforeAll(): Unit = {
    super.beforeAll()

    val bufferedSource = io.Source.fromFile("src/test/resources/data.csv")


    val XBuffer = new ListBuffer[Double]()
    val columnLength = 20


    val lines = bufferedSource.getLines().toArray
    for (i <- lines.indices) {

      val line = lines(i)
      val cols = line.split(",").map(_.trim)

      if(!String.valueOf(cols(0)).equals("label")){

        for(i <- 1 until cols.length){
          XBuffer += String.valueOf(cols(i)).toDouble
        }

      }
      // do whatever you want with the columns here
      println(s"${cols(0)}|${cols(1)}|${cols(2)}|${cols(3)}|${cols(4)}|${cols(5)}|${cols(6)}|${cols(7)}|${cols(8)}|${cols(9)}|${cols(10)}|${cols(11)}|${cols(12)}|${cols(13)}|${cols(14)}|${cols(15)}|${cols(16)}|${cols(17)}|${cols(18)}|${cols(19)}|${cols(20)}|${cols(20)}")
    }
    bufferedSource.close

    X = new DenseMatrix[Double](lines.length -1 , columnLength, XBuffer.toArray, 0, columnLength,true)


  }




}
