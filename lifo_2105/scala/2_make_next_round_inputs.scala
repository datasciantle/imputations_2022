///usr/bin/spark-shell --master yarn --deploy-mode client --driver-memory 12G --packages org.postgresql:postgresql:9.4-1206-jdbc42 --conf spark.maximizeResourceAllocation=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.instances=1000 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.autoBroadcastJoinThreshold=20971520 --conf spark.dynamicAllocation.maxExecutors=1000 --conf "spark.memory.storageFraction=0.1" --conf "spark.memory.fraction=0.1" //--jars "s3committer-0.5.5.jar"

import spark.implicits._
import org.apache.spark.sql.expressions.Window
import org.apache.hadoop.fs.{FileSystem, Path}


// case class Params (
//   val complete_attribute_matrix: String = "s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix",
//   val iteration_imputation_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/results/",
//   val iteration_model_plan: String =  "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/simple_model_plan",
//   val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105/")


// case class Params (
//   val complete_attribute_matrix: String = "s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix",
//   val iteration_imputation_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_1/results/",
//   val iteration_model_plan: String =  "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/simple_model_plan",
//   val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105/")


// case class Params (
//   val complete_attribute_matrix: String = "s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix",
//   val iteration_imputation_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_2/results/",
//   val iteration_model_plan: String =  "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/simple_model_plan",
//   val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105/")

// case class Params (
//   val complete_attribute_matrix: String = "s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix",
//   val iteration_imputation_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105_tuner/iteration_1/results/",
//   val iteration_model_plan: String =  "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/simple_model_plan",
//   val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105_tuner/")

case class Params (
  val complete_attribute_matrix: String = "s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix",
  val iteration_imputation_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105_tuner/iteration_2/results/",
  val iteration_model_plan: String =  "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/simple_model_plan",
  val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105_tuner/")



val params =  new Params

val get_model = udf((x: String) => x.split("/").last.split("_predictions")(0))

val next_round = params.iteration_imputation_results.split("iteration_").last.split("/")(0).toInt + 1

val complete_attribute_matrix = spark.read.load(params.complete_attribute_matrix)
val model_plan = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_model_plan)
val ignore_these = model_plan.where(not($"needs_imputation") and $"proposed_round"===0).select("attribute").distinct.as[String].collect()


val result_files = FileSystem.get(new java.net.URI(params.iteration_imputation_results), sc.hadoopConfiguration).listStatus(new Path(params.iteration_imputation_results)).map(x => x.getPath.toString)
val attributes = result_files.map(x => {
  x.split("/").last
})
val files = result_files.map(x => {
  val att = x.split("/").last
  x + "/" + att + "_predictions"
})

val imputed_data = spark.read.option("header",true).option("inferSchema",true).csv(files.toSeq:_ *)//.withColumn("attribute_model", get_model(input_file_name()))
// val imputed_data = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation_results + "/*/*predictions")//.withColumn("attribute_model", get_model(input_file_name()))
val n_imputed_attributes = imputed_data.select($"attribute_model").distinct.count


val filtered_results = imputed_data.where(not($"attribute_model".isin(ignore_these.toSeq: _*)))
val n_imputed_attributes_filtered = filtered_results.select($"attribute_model").distinct.count

val input_to_next_round_matrix_1 = filtered_results.groupBy("uid","set_uid").pivot("attribute_model").agg(max($"selected_value".cast("int")))
val input_to_next_round_matrix = complete_attribute_matrix.join(input_to_next_round_matrix_1, Seq("uid","set_uid")).cache
input_to_next_round_matrix.count
input_to_next_round_matrix.repartition(1).write.mode("overwrite").save(params.outDir + "iteration_" + next_round + "/inputs/imputed_results_from_previous_round")
input_to_next_round_matrix.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "iteration_" + next_round + "/inputs/imputed_results_from_previous_round_csv")


val xx = spark.read.load("s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_responses_libsvm_1103SS_9551MS_10654TotalAttributes/")





s3://datasci-scantle/unified_ncs/experiments/2105/iteration_1/inputs/imputed_results_from_previous_round