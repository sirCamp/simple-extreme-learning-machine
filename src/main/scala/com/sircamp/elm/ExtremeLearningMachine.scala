package com.sircamp.elm

import breeze.linalg.{DenseMatrix, DenseVector, argmax}

object ExtremeLearningMachine {

  final val SIGMOID = "sigmoid"
  final val HARD_SIGMOID = "hardSigmoid"
  final val TANH = "tanh"
  final val RELU = "relu"
  final val LEAKY_RELU = "leakyRelu"
  final val EXPONENTIAL = "exponential"
  final val ELU = "elu"
  final val SOFT_PLUS = "softPlus"
  final val SOFT_SIGN = "softSign"

  final val availableActivationFunction = Seq(SIGMOID,HARD_SIGMOID,TANH,RELU,LEAKY_RELU,EXPONENTIAL,ELU,SOFT_PLUS,SOFT_SIGN)


}


class ExtremeLearningMachine(val inputLength:Int, val hiddenUnits:Int) extends Serializable {




  private var weights: DenseMatrix[Double] = _

  private var weightedOutput: DenseMatrix[Double] = _

  private var classesNumber:Int = _

  private var activationFunction:String = ExtremeLearningMachine.RELU
  

  def this(inputLength:Int) = {
    this(inputLength, inputLength)

  }


  def getActivationFunction: String = this.activationFunction

  def setActivationFunction(activationFunction:String):Unit = {
    if(ExtremeLearningMachine.availableActivationFunction.contains(activationFunction)){
      this.activationFunction = activationFunction
    }
    else{
      throw new Exception("Non valid activation function")
    }

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

    weightedInput.map(element => {

      var res:Double = 0
      if(element > 0){
        res = element
      }

      res

    })


  }


  def fit(x: DenseMatrix[Double], y:DenseMatrix[Double]) : DenseMatrix[Double] = {


    if(weights == null){
      throw new IllegalArgumentException("Weights must be initialized before fitting the model")
    }

    classesNumber = y.cols

    val hiddenData = this.inputToHidden(x)

    val transposedHiddenData = hiddenData.t

    //np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
    val leftOperand: DenseMatrix[Double] = breeze.linalg.inv( transposedHiddenData * hiddenData )
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
