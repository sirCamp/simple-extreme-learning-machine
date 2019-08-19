package com.sircamp.utils

import breeze.linalg.{BitVector, DenseVector}

object Metrics {

  def accuracy_score(yTrue:DenseVector[Int], yPred:DenseVector[Int]): Double ={

    var result:DenseVector[Boolean] = (yTrue :== yPred).toDenseVector
    var rightValuesCount = 0
    result.foreach(value => {
      value match {
        case true => rightValuesCount += 1
        case false => None
      }
    })

    rightValuesCount / result.size

  }

}
