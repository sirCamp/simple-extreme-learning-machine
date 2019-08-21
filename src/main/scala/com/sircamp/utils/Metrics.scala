package com.sircamp.utils

import breeze.linalg.{DenseVector, sum}


object Metrics {

  private val reduceFunction = (el1:Int, el2:Int) => {
    el1 + el2
  }

  private val mapFunction = (el:Int) => {
    if(el != 0){
      1
    }
    else{
      el
    }
  }

  @throws(classOf[IllegalArgumentException])
  def accuracyScore(yTrue:DenseVector[Int], yPred:DenseVector[Int]): Double = {

    checkDims(yTrue, yPred)
    val correctedClassifiedElements: DenseVector[Int] = ((yTrue - yPred) :== 0).toDenseVector.map(if (_) 1 else 0)
    sum(correctedClassifiedElements).toDouble / yTrue.size.toDouble

  }

  private def checkDims(yTrue:DenseVector[Int], yPred:DenseVector[Int]): Unit = {

    if(yTrue.size != yPred.size){
      throw new IllegalArgumentException(s"yTrue and yPred must have the same dimension, but ${yTrue.size} != ${yPred.size}")
    }
  }

  @throws(classOf[IllegalArgumentException])
  def precisionScore(yTrue:DenseVector[Int], yPred:DenseVector[Int]): Double = {

    val matrixValues = computeConfusionMatrixValues(yTrue, yPred)
    val truePositive = matrixValues._1
    val falsePositive = matrixValues._3

    truePositive.toDouble / (truePositive + falsePositive).toDouble
  }

  @throws(classOf[IllegalArgumentException])
  def recallScore(yTrue:DenseVector[Int], yPred:DenseVector[Int]): Double = {

    val matrixValues = computeConfusionMatrixValues(yTrue, yPred)
    val truePositive = matrixValues._1
    val falseNegative = matrixValues._4

    truePositive.toDouble / (truePositive + falseNegative).toDouble
  }

  @throws(classOf[IllegalArgumentException])
  def f1Score(yTrue:DenseVector[Int], yPred:DenseVector[Int]): Double = {

    val recall = recallScore(yTrue, yPred)
    val precision = precisionScore(yTrue, yPred)

    2 * (precision * recall) / (precision + recall)
  }

  @throws(classOf[IllegalArgumentException])
  private def computeConfusionMatrixValues(yTrue:DenseVector[Int], yPred:DenseVector[Int]):(Int, Int, Int, Int) = {
    checkDims(yTrue, yPred)

    val trueAndPredicted:DenseVector[Int] = yTrue * yPred


    val truePositive:Int  = trueAndPredicted.map( mapFunction ).reduce( reduceFunction )
    val yPredictedSum:Int  = yTrue.map( mapFunction ).reduce( reduceFunction )
    val yTrueSum:Int  = yPred.map( mapFunction ).reduce( reduceFunction )

    val falsePositive = yPredictedSum - truePositive
    val falseNegative = yTrueSum - truePositive
    val trueNegative = yTrue.size - truePositive - falsePositive - falseNegative


    (truePositive, trueNegative, falsePositive, falseNegative)
  }
}
