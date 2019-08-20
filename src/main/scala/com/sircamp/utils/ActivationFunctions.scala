package com.sircamp.utils

import breeze.linalg.DenseMatrix
import breeze.numerics.{abs, exp, log}

object ActivationFunctions {

  val linear: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {

    matrix.map(element => {

      element

    })
  }

  val sigmoid: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {

    matrix.map(element => {

      1/ (1+  exp( -1*element)  )


    })

  }


  val hardSigmoid: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {


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

  val exponential: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {


    matrix.map(element => {

        exp(element)
    })
  }

  val tanh: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {


    matrix.map(element => {
      2/(1 + exp(-2 * element))-1

    })

  }

  val softPlus: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {

    matrix.map(element => {
      log(exp(element) +1 )
    })
  }


  val softSign: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) =>{

    matrix.map(element => {
      element / (abs(element) +1 )
    })

  }

  val relu: DenseMatrix[Double] => DenseMatrix[Double] = (matrix: DenseMatrix[Double]) => {

    matrix.map(element => {

      var res: Double = 0
      if (element > 0) {
        res = element
      }

      res

    })
  }

  val leakyReLu: Double => DenseMatrix[Double] => DenseMatrix[Double] = (alpha:Double) => {

    (matrix: DenseMatrix[Double]) => {
      matrix.map(element => {

        var res: Double = 0
        if (element >= 0) {
          res = element
        }
        else {
          res = alpha * element
        }

        res

      })
    }
  }


  val elu: Double => DenseMatrix[Double] => DenseMatrix[Double] = (alpha:Double) => {
    (matrix: DenseMatrix[Double]) => {
      matrix.map(element => {

        var res: Double = 0
        if (element >= 0) {
          res = element
        }
        else {
          res = alpha * (exp(element) - 1)
        }

        res

      })
    }
  }

}
