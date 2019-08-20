package com.sircamp.elm

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import com.sircamp.utils.ActivationFunctions

class ExtremeLearningMachine(val inputLength:Int, val hiddenUnits:Int) extends Serializable {

  private var weights: DenseMatrix[Double] = _

  private var weightedOutput: DenseMatrix[Double] = _

  private var classesNumber:Int = _

  private var activationFunction:DenseMatrix[Double] => DenseMatrix[Double] = ActivationFunctions.relu

  private var usePseudoInverse = false

  def this(inputLength:Int) = {
    this(inputLength, inputLength)

  }

  def getUsePseudoInverse: Boolean = this.usePseudoInverse

  def setUsePseudoInverse(usePseudoInverse:Boolean):Unit = {

    this.usePseudoInverse = usePseudoInverse
  }

  def getActivationFunction: DenseMatrix[Double] => DenseMatrix[Double] = this.activationFunction

  def setActivationFunction(activationFunction:DenseMatrix[Double] => DenseMatrix[Double]):Unit = {

    this.activationFunction = activationFunction
  }

  def getWeights: DenseMatrix[Double] = this.weights

  def setWeights(weights:DenseMatrix[Double]):Unit = {
    this.weights = weights
  }

  def getWeightedOutput: DenseMatrix[Double] = this.weightedOutput

  def initializeWeights() : Unit = {

    val uniform = breeze.stats.distributions.Uniform(0, 1)
    weights = DenseMatrix.rand(inputLength, hiddenUnits, uniform)

  }


  def inputToHidden(x: DenseMatrix[Double]): DenseMatrix[Double] = {

    // dot product in breeze
    var weightedInput:DenseMatrix[Double] = x * weights

    activationFunction(weightedInput)

  }


  def fit(x: DenseMatrix[Double], y:DenseMatrix[Double]) : DenseMatrix[Double] = {


    if(weights == null){
      throw new IllegalArgumentException("Weights must be initialized before fitting the model")
    }

    classesNumber = y.cols

    val hiddenData = this.inputToHidden(x)

    val transposedHiddenData = hiddenData.t

    //np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
    val leftOperand = usePseudoInverse match {
      case true => breeze.linalg.pinv( transposedHiddenData * hiddenData )
      case false => breeze.linalg.inv( transposedHiddenData * hiddenData )
    }

    val rightOperand: DenseMatrix[Double] = transposedHiddenData * y

    weightedOutput = leftOperand * rightOperand

    weightedOutput
  }

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] = {

    val hiddenData = this.inputToHidden(x)

    val tmp =  hiddenData * weightedOutput

    tmp

  }

  def predictClasses(x: DenseMatrix[Double]):DenseVector[Int] = {

    var rawPredictions = predict(x)

    var predictions = DenseVector.zeros[Int](rawPredictions.rows)

    for(j <- 0 until rawPredictions.rows){

      val maxIndex = argmax(rawPredictions(j,::))

      predictions(j) = maxIndex
    }

    predictions
  }


}
