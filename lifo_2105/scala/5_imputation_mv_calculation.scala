///usr/bin/spark-shell --master yarn --deploy-mode client --driver-memory 12G --packages org.postgresql:postgresql:9.4-1206-jdbc42 --conf spark.maximizeResourceAllocation=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.instances=1000 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.autoBroadcastJoinThreshold=20971520 --conf spark.dynamicAllocation.maxExecutors=1000 --conf "spark.memory.storageFraction=0.1" --conf "spark.memory.fraction=0.1" //--jars "s3committer-0.5.5.jar"

import spark.implicits._
import org.apache.spark.sql.expressions.Window
import scala.collection.mutable.WrappedArray
import org.apache.hadoop.fs.{FileSystem, Path}


case class Params (
  val focal_bsm: String =  "s3://datasci-scantle/unified_ncs/2105/inputs/focal_bsm/*",
  val complete_taxonomy_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed",
  val distributions_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions",
  val valid_respondents: String =  "s3://datasci-scantle/unified_ncs/2105/bsm/data/validRespondents",
  val iteration_imputation_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105_tuner/iteration_1/results/",
  val randomly_chosen_audiences: String = "s3://datasci-scantle/unified_ncs/experiments/2105/summaries/multivariate_distributions/randomly_chosen_audiences",
  val randomly_chosen_svkey_avkey: String = "s3://datasci-scantle/unified_ncs/experiments/2105/summaries/multivariate_distributions/randomly_chosen_svkey_avkey",
  val randomly_chosen_platform_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/summaries/multivariate_distributions/randomly_chosen_platform_results/*",
  val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105_tuner/")


val params =  new Params
val get_model = udf((x: String) => x.split("/").last.split("_predictions")(0))
val avg_array = udf((x: WrappedArray[Double]) => x.toArray.sum / x.size)
val get_attribute_value = udf((x: String) => x.split("_")(0).toInt)
val get_set_value = udf((x: String) => x.split("_").last)


val iteration = params.iteration_imputation_results.split("iteration_").last.split("/")(0)


val focal_bsm = spark.read.load(params.focal_bsm)
val complete_taxonomy_df = spark.read.load(params.complete_taxonomy_df)
val distributions_df = spark.read.load(params.distributions_df)
val valid_respondents = spark.read.load(params.valid_respondents).select("uid","wavekey","respondentweight").repartition(1).cache
val randomly_chosen_audiences = spark.read.parquet(params.randomly_chosen_audiences)
val randomly_chosen_svkey_avkey = spark.read.load(params.randomly_chosen_svkey_avkey)
val randomly_chosen_platform_results = spark.read.parquet(params.randomly_chosen_platform_results)

val result_files = FileSystem.get(new java.net.URI(params.iteration_imputation_results), sc.hadoopConfiguration).listStatus(new Path(params.iteration_imputation_results)).map(x => x.getPath.toString)
val attributes = result_files.map(x => {
  x.split("/").last
})


val distributions = complete_taxonomy_df.select("sakey","svkey","multi","attribute","alternate_svkey_binary_single","value","is_display_logic_sakey","is_display_logic_svkey","min_level").
  join(distributions_df, Seq("sakey","svkey","multi"), "inner").distinct.cache
val waveweights = valid_respondents.groupBy("wavekey").agg(sum($"respondentweight").cast("double") as "wave_total_weight").repartition(1).cache
val taxonomy = distributions.select($"sakey",$"svkey",$"multi",$"attribute", when($"value".isNull, 1).otherwise($"value" - 1) as "attribute_value").distinct.repartition(1).cache
val waves = valid_respondents.select("wavekey").distinct.as[String].collect.sorted

// taxonomy.join(imputation, $"attribute"===$"attribute_model" and $"attribute_value"===$"selected_value").show
val platform_results = randomly_chosen_platform_results.withColumn("attributes", concat_ws("__",$"attribute1",$"attribute2")).withColumn("sakeys", concat_ws("__",$"sakey_1",$"sakey_2")).withColumn("svkeys", concat_ws("__",$"svkey_1",$"svkey_2")).withColumn("custom_audience_expression", concat_ws("-",$"attributes",$"sakeys",$"svkeys"))
val random_audience_attribute_keys = platform_results.select("custom_audience_expression").distinct.as[String].collect()



val donzo = random_audience_attribute_keys.par.map(aud => {
  // val aud = random_audience_attribute_keys(0)
  val aud_result_path = params.outDir + "multivariate_distributions/raw/iteration_" + iteration + "/" + aud
  val resultExists = FileSystem.get(new java.net.URI(aud_result_path), sc.hadoopConfiguration).exists(new Path(aud_result_path))
  if(!resultExists){
    println(aud_result_path)
    val attributes = aud.split("-")(0)
    val sakeys = aud.split("-")(1)
    val svkeys = aud.split("-")(2)
    val att1 = attributes.split("__")(0)
    val att2 = attributes.split("__")(1)
    val sakey1 = sakeys.split("__")(0)
    val sakey2 = sakeys.split("__")(1)
    val svkey1 = svkeys.split("__")(0)
    val svkey2 = svkeys.split("__")(1)
    val df1 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation_results + att1 + "/" + att1 + "_predictions").withColumn("source", lit("att1"))
    val df2 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation_results + att2 + "/" + att2 + "_predictions").withColumn("source", lit("att2"))
    val data = df1.union(df2)
    val focal_data_1 = taxonomy.where($"svkey".isin(svkeys.split("__").map(_.toInt).toSeq: _*)).join(data.drop("multi"), $"attribute"===$"attribute_model" and $"attribute_value"===$"selected_value")
    val focal_data_2 = focal_data_1.groupBy("uid","wavekey").pivot($"source", Array("att1","att2")).agg(max($"selected_value")).where($"att1".isNotNull and $"att2".isNotNull)
    val focal_data_3 = focal_data_2.join(valid_respondents, Seq("uid","wavekey"), "inner")
    val focal_data_4 = focal_data_3.groupBy("wavekey").agg(sum($"respondentweight") as "wave_audience_weight").join(waveweights, "wavekey").withColumn("audience_imputed_distribution", $"wave_audience_weight"/$"wave_total_weight")
    val focal_data = focal_data_4.withColumn("custom_audience_expression",lit(aud)).
      withColumn("attribute1", lit(att1)).withColumn("attribute2", lit(att2)).withColumn("sakey_1", lit(sakey1)).withColumn("sakey_2", lit(sakey2)).withColumn("svkey_1", lit(svkey1)).withColumn("svkey_2", lit(svkey2)).
      groupBy("custom_audience_expression","attribute1","attribute2","sakey_1","sakey_2","svkey_1","svkey_2").pivot("wavekey",waves).agg(max($"audience_imputed_distribution")).na.fill(0.0, waves).
      withColumn("average_imputed_distribution", avg_array(array(waves.toSeq.map(col(_)): _*))).withColumn("iteration", lit(iteration))
    focal_data.repartition(1).write.mode("overwrite").save(aud_result_path)
  }
  true
})


val df1 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation_results.replace("iteration_1","iteration_2") + "198820_285015/198820_285015_predictions")



df1.join(focal_bsm.where($"sakey"===129404 and $"label"===1.0).select($"uid",$"wavekey",$"svkey" as "gender"),"uid").groupBy("selected_value").pivot("gender").agg(countDistinct($"uid")).show






