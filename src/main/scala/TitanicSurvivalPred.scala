import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.functions._

object TitanicSurvivalPred extends App{

  val spark = SparkSession.builder()
    .appName("Titanic Survival Prediction")
    .master("local[*]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  // Load the training and test datasets
  val trainDF = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("D:/Scala/Spark_ML/src/test/resources/train.csv")

  val testData = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("D:/Scala/Spark_ML/src/test/resources/test.csv")

  // Perform exploratory data analysis on the training dataset
  trainDF.printSchema()
  trainDF.show(20)

  // Calculate the mean age
  val meanAge = trainDF.select(mean("age")).head().getDouble(0).round.toInt

  // Fill null values in "age" column with the mean age
  val trainData = trainDF.na.fill(meanAge, Seq("age"))
  trainData.show(20)

  println("Total No Of Rows:" + trainData.count())
  trainData.describe().show()
  //trainDF.describe("Age", "SibSp", "Parch", "Fare").show()


  //count number of passengers based on sex
  trainData.groupBy("Sex").count().show(false)

  //count sum of total ticket fare by class type
  trainData.groupBy("Pclass").sum("Fare").show(false)

  //trainData.groupBy("Sex","Age")
  val children = sum(when(col("age") < 19, 1).otherwise(0)).as("children_count")
  val adults = sum(when(col("age").between(19, 60), 1).otherwise(0)).as("adults_count")
  val seniorcitizen = sum(when(col("age") > 60, 1).otherwise(0)).as("seniorcitizen_count")

  println("children" + children)
  //No of passengers grouped by age
  val result = trainData.select(children, adults, seniorcitizen)
  result.show()

  //no of passengers grouped by age and gender
  trainData.groupBy("sex").agg(children, adults, seniorcitizen).show()

  val data = trainData
    .withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    .withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))
    .drop("PassengerId","Embarked", "Fare","Cabin")

  data.show(20)


}
