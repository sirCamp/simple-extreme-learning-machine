import java.nio.file.{Files, Paths}

import breeze.linalg.{DenseMatrix, DenseVector}
import com.sircamp.elm.ExtremeLearningMachine
import com.sircamp.utils.{ActivationFunctions, Metrics}
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

@RunWith(classOf[JUnitRunner])
class ExtremeLearningMachineSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  val filePath = "src/test/resources/elmSaved.model"


  var y : DenseMatrix[Double] = _

  var X : DenseMatrix[Double] = _

  var yPlain : DenseVector[Int] = _


  var yTrain : DenseMatrix[Double] = _

  var yTest : DenseMatrix[Double] = _

  var XTrain : DenseMatrix[Double] = _

  var XTest : DenseMatrix[Double] = _



  override protected def beforeAll(): Unit = {
    super.beforeAll()

    val bufferedSource = io.Source.fromFile("src/test/resources/data.csv")

    val yBuffer = new ListBuffer[Double]()
    val XBuffer = new ListBuffer[Double]()
    val columnLength = 20


    val lines = bufferedSource.getLines().toArray
    for (i <- lines.indices) {

      val line = lines(i)
      val cols = line.split(",").map(_.trim)

      if(!String.valueOf(cols(0)).equals("label")){


        yBuffer += String.valueOf(cols(0)).toDouble
        for(i <- 1 until cols.length){
          XBuffer += String.valueOf(cols(i)).toDouble
        }

      }
      // do whatever you want with the columns here
      println(s"${cols(0)}|${cols(1)}|${cols(2)}|${cols(3)}|${cols(4)}|${cols(5)}|${cols(6)}|${cols(7)}|${cols(8)}|${cols(9)}|${cols(10)}|${cols(11)}|${cols(12)}|${cols(13)}|${cols(14)}|${cols(15)}|${cols(16)}|${cols(17)}|${cols(18)}|${cols(19)}|${cols(20)}|${cols(20)}")
    }
    bufferedSource.close

    yPlain = new DenseVector[Int](yBuffer.result().map(_.toInt).toArray)(90 until 100)

    val classesNumber = yBuffer.toSet.size

    val mapping = new mutable.HashMap[Double, Int]()
    val ySet = yBuffer.toSet[Double].toList
    for(i <- ySet.indices){

      mapping.put(ySet(i), i)

    }

    val yList: List[Array[Double]] = yBuffer.result().map(elm => {

      val arr = Array.fill(n=classesNumber){0.0}
      arr(mapping(elm)) = 1

      arr
    })

    DenseMatrix.zeros[Double](yBuffer.length, classesNumber)

    y = new DenseMatrix[Double](yBuffer.length, classesNumber, yList.toArray.flatten, 0, 2, true)
    X = new DenseMatrix[Double](yBuffer.length, columnLength, XBuffer.toArray, 0, columnLength,true)

    yTrain = y(0 until 90, ::)
    yTest = y(90 until 100, ::)

    XTrain = X(0 until 90, ::)
    XTest = X(90 until 100, ::)

  }

  override protected def afterAll(): Unit = {

    super.afterAll()
    if(Files.exists(Paths.get(filePath))){
      Files.delete(Paths.get(filePath))
    }

  }

  test("Test simple constructor") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)

    var assertionList = ListBuffer[Assertion]()

    assertionList += assert(elm.inputLength == X.cols)

    assertionList += assert(elm.hiddenUnits == X.cols)

    assertionList.result().forall(_ == Succeeded)


  }

  test("Test inputLength constructor") {

    val elm = new ExtremeLearningMachine(X.cols)

    assert(elm.inputLength == X.cols)

  }


  test("Test setWeights") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)

    val uniform = breeze.stats.distributions.Uniform(0, 1)
    val weights = DenseMatrix.rand(X.cols, X.cols, uniform)

    elm.setWeights(weights)

    elm.getWeights.toArray should contain theSameElementsAs weights.toArray


  }

  test("Test initializeWeights") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    assert(elm.getWeights != null)

  }

  test("Test fit") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    val outWeights = elm.fit(X, y)

    assert(outWeights != null)

  }


  test("Test predict") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.fit(XTrain, yTrain)

    elm.predictClasses(XTrain)

    assert(elm.getWeights != null)

  }

  test("Test predict classes") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(1.0 == accuracy)

  }

  test("Test save model") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.hardSigmoid)

    elm.setUsePseudoInverse(true)

    elm.fit(XTrain, yTrain)

    val bytes = elm.save(filePath)

    var assertions = new ListBuffer[Assertion]()

    assertions += assert(Files.exists(Paths.get(filePath)))
    assertions += Files.readAllBytes(Paths.get(filePath)) should contain theSameElementsAs bytes

    assertions.forall(_ == Succeeded)

  }


  test("Test load model") {


    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.hardSigmoid)

    elm.setUsePseudoInverse(true)

    elm.fit(XTrain, yTrain)

    elm.save(filePath)

    val loadedModel = ExtremeLearningMachine.load(filePath)

    assert(elm.equals(loadedModel))

  }

  test("Test predict classes with leakyRelu activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.leakyReLu(0.1d))

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(1.0 == accuracy)

  }

  test("Test predict classes with elu activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.elu(0.1d))

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(1.0 == accuracy)

  }

  test("Test predict classes with sigmoid activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.sigmoid)

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(accuracy <= 1.0)

  }

  test("Test predict classes with hardSigmoid activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.hardSigmoid)

    elm.setUsePseudoInverse(true)

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(accuracy <= 1.0)

  }

  test("Test predict classes with exponential activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.exponential)

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(accuracy <= 1.0)

  }

  test("Test predict classes with tanh activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.tanh)

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(accuracy <= 1.0)

  }


  test("Test predict classes with softPlus activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.softPlus)

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)
    println("Accuracy: "+accuracy)

    assert(accuracy <= 1.0)

  }

  test("Test predict classes with softSign activation function") {

    val elm = new ExtremeLearningMachine(X.cols, X.cols)
    elm.initializeWeights()

    elm.setActivationFunction(ActivationFunctions.softSign)

    elm.fit(XTrain, yTrain)

    val yPred = elm.predictClasses(XTest)

    val accuracy = Metrics.accuracyScore(yPlain,yPred)

    println("Accuracy: "+accuracy)

    assert(accuracy <= 1.0)

  }


}