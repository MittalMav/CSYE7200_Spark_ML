import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer,VectorIndexer,VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.functions._


object TitanicPrediction {

  def main(args: Array[String]): Unit = {

    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("Titanic Survival Prediction")
      .master("local[*]")
      .getOrCreate()

    // Load the training and test datasets
    val trainData = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("D:/Scala/Spark_ML/src/test/resources/train.csv")

    val testData = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("D:/Scala/Spark_ML/src/test/resources/test.csv")

    // Perform exploratory data analysis on the training dataset
    trainData.describe().show()

    // Feature engineering - create new attributes and remove unnecessary columns
    val data = trainData
      .withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
      .withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))
      .drop("PassengerId", "Name", "Ticket", "Cabin")

    data.show(5)
    // Convert categorical variables into numerical variables using one-hot encoding
    val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("sexIndexer").fit(data)
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("embarkedIndexer").fit(data)
//    val sexEncoder = new OneHotEncoder().setInputCol("sexIndexer").setOutputCol("SexVec").fit(data)
  //  val embarkedEncoder = new OneHotEncoder().setInputCol("embarkedIndexer").setOutputCol("EmbarkedVec").fit(data)

    // Prepare the data for training
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkedVec", "FamilySize", "IsAlone"))
      .setOutputCol("features")

    // Split the training dataset into training and validation sets
    val Array(trainingData, validationData) = data.randomSplit(Array(0.8, 0.2))

    // Create a logistic regression model
    val lr = new LogisticRegression()

    // Create a pipeline to chain the stages together
    val pipeline = new Pipeline()
      .setStages(Array(sexIndexer, embarkedIndexer, assembler, lr))

    // Train the model on the training dataset
    val model = pipeline.fit(trainingData)

    // Use the trained model to predict the survival status of passengers in the test dataset
   // val predictions = model.transform(testData)

//    // Evaluate the performance of the model
//    val evaluator = new BinaryClassificationEvaluator().setLabelCol("Survived")
//    val auc = evaluator.evaluate(predictions)
//    println(s"Area under ROC curve = $auc")
//
//    // Show the predicted survival status of passengers in the test dataset
//    predictions.select("PassengerId", "prediction").show()
//
//    // Stop the Spark session
//    spark.stop()
  }

}
