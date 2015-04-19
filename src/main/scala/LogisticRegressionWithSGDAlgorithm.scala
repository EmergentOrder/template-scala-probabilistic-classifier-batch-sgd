package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class AlgorithmParams(
  maxIter: Int,
  regParam: Double,
  stepSize: Double
) extends Params

// extends P2LAlgorithm because the MLlib's LogisticRegressionModel doesn't contain RDD.
class LogisticRegressionWithSGDAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, LogisticRegressionModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): LogisticRegressionModel = {
    // MLLib cannot handle empty training data.
    require(!data.labeledPoints.take(1).isEmpty,
      s"RDD[labeldPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
   
    val lr = new LogisticRegressionWithSGD()
    lr.optimizer.setStepSize(ap.stepSize).setNumIterations(ap.maxIter).setRegParam(ap.regParam)
  
    lr.run(data.labeledPoints)
  }

  def predict(model: LogisticRegressionModel, query: Query): PredictedResult = {
    model.clearThreshold()
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }

}
