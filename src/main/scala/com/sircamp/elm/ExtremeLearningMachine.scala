package com.sircamp.elm

import java.io._
import java.nio.file.{Files, Paths}

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import com.sircamp.utils.ActivationFunctions

object ExtremeLearningMachine {

  @throws(classOf[IOException])
  @throws(classOf[ClassCastException])
  def load(path:String): ExtremeLearningMachine = {

    val bytes = Files.readAllBytes(Paths.get(path))
    val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
    val value = ois.readObject.asInstanceOf[ExtremeLearningMachine]
    ois.close()
    value
  }

}

class ExtremeLearningMachine(val inputLength:Int, val hiddenUnits:Int) extends Serializable {

  private var weights: DenseMatrix[Double] = _

  private var weightedOutput: DenseMatrix[Double] = _

  private var classesNumber:Int = _

  private var activationFunction:DenseMatrix[Double] => DenseMatrix[Double] = ActivationFunctions.relu

  private var usePseudoInverse = false

  def this(inputLength:Int) = {
    this(inputLength, inputLength)

  }

  @throws(classOf[IOException])
  def save(path:String): Array[Byte] = {

    val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    oos.writeObject(this)
    stream.toByteArray

    val filePath = Paths.get(path)
    Files.write(filePath, stream.toByteArray)

    oos.close()
    stream.toByteArray
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

  def fit(x: DenseMatrix[Double], y:DenseMatrix[Double]) : DenseMatrix[Double] = {


    if(weights == null){
      throw new IllegalArgumentException("Weights must be initialized before fitting the model")
    }

    classesNumber = y.cols

    val hiddenData = this.inputToHidden(x)

    val transposedHiddenData = hiddenData.t

    val leftOperand = if (usePseudoInverse) {
      breeze.linalg.pinv(transposedHiddenData * hiddenData)
    } else {
      breeze.linalg.inv(transposedHiddenData * hiddenData)
    }

    val rightOperand: DenseMatrix[Double] = transposedHiddenData * y

    weightedOutput = leftOperand * rightOperand

    weightedOutput
  }

  def inputToHidden(x: DenseMatrix[Double]): DenseMatrix[Double] = {

    // dot product in breeze
    val weightedInput:DenseMatrix[Double] = x * weights

    activationFunction(weightedInput)

  }

  def predictClasses(x: DenseMatrix[Double]):DenseVector[Int] = {

    val rawPredictions = predict(x)

    val predictions = DenseVector.zeros[Int](rawPredictions.rows)

    for(j <- 0 until rawPredictions.rows){

      val maxIndex = argmax(rawPredictions(j,::))

      predictions(j) = maxIndex
    }

    predictions
  }

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] = {

    if(weightedOutput == null){
      throw new UnsupportedOperationException("Model mast be fit before to predict")
    }
    val hiddenData = this.inputToHidden(x)

    hiddenData * weightedOutput

  }

  override def equals(other: Any): Boolean = other match {
    case that: ExtremeLearningMachine =>
      (that canEqual this) &&
        weights.equals(that.weights) &&
        weightedOutput.equals(that.weightedOutput) &&
        classesNumber == that.classesNumber &&
        usePseudoInverse == that.usePseudoInverse &&
        inputLength == that.inputLength &&
        hiddenUnits == that.hiddenUnits
    case _ => false
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[ExtremeLearningMachine]

}
