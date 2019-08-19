package com.sircamp.utils

import breeze.linalg.DenseMatrix
import breeze.numerics.{abs, exp, log}

object ActivationFunctions {

  def linear(matrix: DenseMatrix[Double]):DenseMatrix[Double] = {

    matrix.map(element => {

      element

    })
  }

  def sigmoid(matrix: DenseMatrix[Double]):DenseMatrix[Double] = {

    matrix.map(element => {

      1/ (1+  exp( -1*element)  )


    })

  }


  def hardSigmoid(matrix: DenseMatrix[Double]):DenseMatrix[Double] = {


      matrix.map(element => {

        var res = 0.0
        if(element < -2.5){
          res = 0.0
        }
        else if(element > 2.5){
          res = 1.0
        }
        else{

          //`-2.5 <= x <= 2.5
          res = 0.2*element + 0.5
        }

        res
      })
  }

  def exponential(matrix: DenseMatrix[Double]):DenseMatrix[Double] = {


    matrix.map(element => {

        exp(element)
    })
  }

  def tanh(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {


    matrix.map(element => {
      2/(1 + exp(-2 * element))-1

    })

  }

  def softPlus(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {

    matrix.map(element => {
      log(exp(element) +1 )
    })
  }


  def softSign(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {

    matrix.map(element => {
      element / (abs(element) +1 )
    })

  }

  def relu(matrix: DenseMatrix[Double]):DenseMatrix[Double] = {

    matrix.map(element => {

      var res: Double = 0
      if (element > 0) {
        res = element
      }

      res

    })
  }

  def leakyReLu(matrix: DenseMatrix[Double], alpha:Double = 0.1d):DenseMatrix[Double] = {

    matrix.map(element => {

      var res: Double = 0
      if (element >= 0) {
        res = element
      }
      else{
        res = alpha * element
      }

      res

    })
  }


  def elu(matrix: DenseMatrix[Double], alpha:Double = 0.1d):DenseMatrix[Double] = {

    matrix.map(element => {

      var res: Double = 0
      if (element >= 0) {
        res = element
      }
      else{
        res = alpha * (exp(element)-1)
      }

      res

    })
  }



}
