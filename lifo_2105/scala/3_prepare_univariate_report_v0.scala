///usr/bin/spark-shell --master yarn --deploy-mode client --driver-memory 12G --packages org.postgresql:postgresql:9.4-1206-jdbc42 --conf spark.maximizeResourceAllocation=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.instances=1000 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.autoBroadcastJoinThreshold=20971520 --conf spark.dynamicAllocation.maxExecutors=1000 --conf "spark.memory.storageFraction=0.1" --conf "spark.memory.fraction=0.1" //--jars "s3committer-0.5.5.jar"

import spark.implicits._
import org.apache.spark.sql.expressions.Window
import scala.collection.mutable.WrappedArray
import org.apache.hadoop.fs.{FileSystem, Path}

// case class Params (
//   val iteration_imputation0_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_2/results/",
//   val focal_bsm: String =  "s3://datasci-scantle/unified_ncs/2105/inputs/focal_bsm/*",
//   val complete_taxonomy_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed",
//   val distributions_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions",
//   val valid_respondents: String =  "s3://datasci-scantle/unified_ncs/2105/bsm/data/validRespondents",
//   val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105/")

case class Params (
  val iteration_imputation0_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_3/results/",
  val focal_bsm: String =  "s3://datasci-scantle/unified_ncs/2105/inputs/focal_bsm/*",
  val complete_taxonomy_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed",
  val distributions_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions",
  val valid_respondents: String =  "s3://datasci-scantle/unified_ncs/2105/bsm/data/validRespondents",
  val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105/")


val params =  new Params


val get_model = udf((x: String) => x.split("/").last.split("_predictions")(0))
val avg_array = udf((x: WrappedArray[Double]) => x.toArray.sum / x.size)
val get_attribute_value = udf((x: String) => x.split("_")(0).toInt)
val get_set_value = udf((x: String) => x.split("_").last)

val iteration = params.iteration_imputation0_results.split("iteration_").last.split("/")(0)

val focal_bsm = spark.read.load(params.focal_bsm)
val complete_taxonomy_df = spark.read.load(params.complete_taxonomy_df)
val distributions_df = spark.read.load(params.distributions_df)
val valid_respondents = spark.read.load(params.valid_respondents)

val result_files = FileSystem.get(new java.net.URI(params.iteration_imputation0_results), sc.hadoopConfiguration).listStatus(new Path(params.iteration_imputation0_results)).map(x => x.getPath.toString)
val attributes = result_files.map(x => {
  x.split("/").last
})


// val imputation0 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation0_results + "{" + attributes.take(5).mkString(",") + "}/{" + attributes.take(5).mkString(",") + "}_predictions").withColumn("attribute_model", get_model(input_file_name()))
val imputation0 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation0_results + "{" + attributes.mkString(",") + "}/{" + attributes.mkString(",") + "}_predictions")
val metrics0 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation0_results + "{" + attributes.mkString(",") + "}/{" + attributes.mkString(",") + "}_metrics")


val distributions = complete_taxonomy_df.select("sakey","svkey","multi","attribute","alternate_svkey_binary_single","value","is_display_logic_sakey","is_display_logic_svkey","min_level").
  join(distributions_df, Seq("sakey","svkey","multi"), "inner").distinct.cache
val waveweights = valid_respondents.groupBy("wavekey").agg(sum($"respondentweight").cast("double") as "wave_total_weight").cache
val taxonomy = distributions.select($"sakey",$"svkey",$"multi",$"attribute", when($"value".isNull, 1).otherwise($"value" - 1) as "attribute_value").distinct.cache
val waves = valid_respondents.select("wavekey").distinct.as[String].collect.sorted

val fs = FileSystem.get(new java.net.URI(params.outDir + "summaries/univariate_distributions/"), sc.hadoopConfiguration)
val distributionsFileExists = fs.exists(new Path(params.outDir + "summaries/univariate_distributions/production_univariate_distributions"))
if(!distributionsFileExists) distributions.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/univariate_distributions/production_univariate_distributions")


val imp0 = imputation0.join(taxonomy, ($"attribute_model" === $"attribute" and $"selected_value" === $"attribute_value"), "inner").
  join(valid_respondents.select("uid","wavekey","respondentweight"), "uid")
val imp0_univariate_distributions = imp0.groupBy("attribute","sakey","svkey","wavekey").agg(sum($"respondentweight") as "wave_attribute_weight").
  join(waveweights,"wavekey").withColumn("wave_distribution", $"wave_attribute_weight"/$"wave_total_weight").groupBy("attribute","sakey","svkey").pivot("wavekey", waves).agg(max($"wave_distribution")).
  withColumn("average_imputed_distribution", avg_array(array(waves.toSeq.map(col(_)): _*))).withColumn("round", lit(0))
val met0 = metrics0.withColumn("value", get_attribute_value($"data_slice")).withColumn("set", get_set_value($"data_slice")).
  join(taxonomy, ($"model" === $"attribute" and $"value" === $"attribute_value")).
  select("attribute","sakey","svkey","multi","set","loss","roc_score").withColumn("round", lit(0))


imp0_univariate_distributions.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/univariate_distributions/iteration_" + iteration)
met0.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/metrics/iteration_" + iteration)
