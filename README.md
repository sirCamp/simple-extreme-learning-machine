[![Build Status](https://travis-ci.com/sirCamp/simple-extreme-learning-machine.svg?branch=master)](https://travis-ci.com/sirCamp/simple-extreme-learning-machine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scala](https://img.shields.io/badge/scala-v2.12-blue)](https://img.shields.io/badge/scala-v2.12-blue)


# A simple implementation of Extreme Learning Machine

This is a simple scala & Breeze implementation of an Extreme Learning Machine.

It's based on Breeze, a library designed to perform with scientific computation, in order to have great performances thanks to the linear algebra optimizations and to the usage of BLAS.

For further information take a look to Breeze github page: [https://github.com/scalanlp/breeze/wiki](https://github.com/scalanlp/breeze/wiki) 

I suggest also to take a look to linear algebra concepts page: [https://github.com/scalanlp/breeze/wiki/Breeze-Linear-Algebra](https://github.com/scalanlp/breeze/wiki/Breeze-Linear-Algebra)


### How to

In order to use take a look to the following example:
```scala
 import com.sircamp.elm.ExtremeLearningMachine


 var featuresLength = 28*28 //MINST dataset
 var hiddenLayerDimension = 1024
 val elm = new ExtremeLearningMachine(featuresLength, hiddenLayerDimension)
 
 /**
  Initialize the weights of the hidden layer with random uniform distribution
  Otherwise you can set the weights by your own. 
  Weights must be a DenseMatrix[Double] where rows are equal to the featuresLength
 **/
 elm.initializeWeights() 

 /**
  fit the model.
  XTrain and yTrain must be DenseMatrix[Double].
  yTrain must be the one hot encoded version of the original label
**/
 elm.fit(XTrain, yTrain)

/**
  predictClasses return a DenseVector[Int] with the index of the predicted class 
**/
 var yPred = elm.predictClasses(XTest)


/**
  predict return a DenseMatrix[Double] where each row contains the probability of the element to belongs to the class 
**/
 var yProbabilityPred = elm.predict(XTest)

 println("Accuracy: "+Metrics.accuracy_score(yPlain,yPred))


```

Set a different Activation function
```scala
import com.sircamp.elm.ExtremeLearningMachine


 var featuresLength = 28*28 //MINST dataset
 var hiddenLayerDimension = 1024
 val elm = new ExtremeLearningMachine(featuresLength, hiddenLayerDimension)
 
 /**
  Initialize the weights of the hidden layer with random uniform distribution
  Otherwise you can set the weights by your own. 
  Weights must be a DenseMatrix[Double] where rows are equal to the featuresLength
 **/
 elm.initializeWeights() 

 /**
   This return the LeakyReLu function with the alpha param
 **/
 elm.setActivationFunction(ActivationFunctions.leakyReLu(0.2))


 /**
   This return the Tanh function
 **/
 elm.setActivationFunction(ActivationFunctions.tanh)
```

For more example take a look to the tests