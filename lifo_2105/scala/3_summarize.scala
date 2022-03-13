///usr/bin/spark-shell --master yarn --deploy-mode client --driver-memory 12G --packages org.postgresql:postgresql:9.4-1206-jdbc42 --conf spark.maximizeResourceAllocation=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.instances=1000 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.autoBroadcastJoinThreshold=20971520 --conf spark.dynamicAllocation.maxExecutors=1000 --conf "spark.memory.storageFraction=0.1" --conf "spark.memory.fraction=0.1" //--jars "s3committer-0.5.5.jar"

import spark.implicits._
import org.apache.spark.sql.expressions.Window
import scala.collection.mutable.WrappedArray

case class Params (
  val iteration_imputation0_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/results/",
  val iteration_imputation1_results: String = "s3://datasci-scantle/unified_ncs/experiments/2105/iteration_1/results/",
  val focal_bsm: String =  "s3://datasci-scantle/unified_ncs/2105/inputs/focal_bsm/*",
  val complete_taxonomy_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed",
  val distributions_df: String =  "s3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions",
  val valid_respondents: String =  "s3://datasci-scantle/unified_ncs/2105/bsm/data/validRespondents",
  val sz_sakey: String = "s3://resonate-guardians-pipeline-qa/data/xwave/tlcm/2105/structural-zero/sakey/",
  val sz_svkey: String = "s3://resonate-guardians-pipeline-qa/data/xwave/tlcm/2105/structural-zero/svkey/",
  val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105/")


val params =  new Params
val get_model = udf((x: String) => x.split("/").last.split("_predictions")(0))
val avg_array = udf((x: WrappedArray[Double]) => x.toArray.sum / x.size)
val get_attribute_value = udf((x: String) => x.split("_")(0).toInt)
val get_set_value = udf((x: String) => x.split("_").last)

val focal_bsm = spark.read.load(params.focal_bsm)
val complete_taxonomy_df = spark.read.load(params.complete_taxonomy_df)
val distributions_df = spark.read.load(params.distributions_df)
val valid_respondents = spark.read.load(params.valid_respondents)

val imputation0 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation0_results + "/*/*predictions").withColumn("attribute_model", get_model(input_file_name()))
val imputation1 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation1_results + "/*/*predictions")
val metrics0 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation0_results + "/*/*metrics")
val metrics1 = spark.read.option("header",true).option("inferSchema",true).csv(params.iteration_imputation1_results + "/*/*metrics")



val distributions = complete_taxonomy_df.select("sakey","svkey","multi","attribute","alternate_svkey_binary_single","value","is_display_logic_sakey","is_display_logic_svkey","min_level").
  join(distributions_df, Seq("sakey","svkey","multi"), "inner").distinct.cache
val waveweights = valid_respondents.groupBy("wavekey").agg(sum($"respondentweight").cast("double") as "wave_total_weight").cache
val taxonomy = distributions.select($"sakey",$"svkey",$"multi",$"attribute", when($"value".isNull, 1).otherwise($"value" - 1) as "attribute_value").distinct.cache
val waves = valid_respondents.select("wavekey").distinct.as[String].collect.sorted


val imp0 = imputation0.join(taxonomy, ($"attribute_model" === $"attribute" and $"selected_value" === $"attribute_value"), "inner").
  join(valid_respondents.select("uid","wavekey","respondentweight"), "uid")
val imp1 = imputation1.join(taxonomy, ($"attribute_model" === $"attribute" and $"selected_value" === $"attribute_value"), "inner").
  join(valid_respondents.select("uid","wavekey","respondentweight"), "uid")


val imp0_univariate_distributions = imp0.groupBy("attribute","sakey","svkey","wavekey").agg(sum($"respondentweight") as "wave_attribute_weight").
  join(waveweights,"wavekey").withColumn("wave_distribution", $"wave_attribute_weight"/$"wave_total_weight").groupBy("attribute","sakey","svkey").pivot("wavekey", waves).agg(max($"wave_distribution")).
  withColumn("average_imputed_distribution", avg_array(array(waves.toSeq.map(col(_)): _*))).withColumn("round", lit(0))


val imp1_univariate_distributions = imp1.groupBy("attribute","sakey","svkey","wavekey").agg(sum($"respondentweight") as "wave_attribute_weight").
  join(waveweights,"wavekey").withColumn("wave_distribution", $"wave_attribute_weight"/$"wave_total_weight").groupBy("attribute","sakey","svkey").pivot("wavekey", waves).agg(max($"wave_distribution")).
  withColumn("average_imputed_distribution", avg_array(array(waves.toSeq.map(col(_)): _*))).withColumn("round", lit(1))


imp0_univariate_distributions.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/univariate_distributions/iteration_0")
imp1_univariate_distributions.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/univariate_distributions/iteration_1")


val met0 = metrics0.withColumn("value", get_attribute_value($"data_slice")).withColumn("set", get_set_value($"data_slice")).
  join(taxonomy, ($"model" === $"attribute" and $"value" === $"attribute_value")).
  select("attribute","sakey","svkey","multi","set","loss","roc_score").withColumn("round", lit(0))

val met1 = metrics1.withColumn("value", get_attribute_value($"data_slice")).withColumn("set", get_set_value($"data_slice")).
  join(taxonomy, ($"model" === $"attribute" and $"value" === $"attribute_value")).
  select("attribute","sakey","svkey","multi","set","loss","roc_score").withColumn("round", lit(1))

met0.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/metrics/iteration_0")
met1.repartition(1).write.mode("overwrite").option("header",true).csv(params.outDir + "summaries/metrics/iteration_1")



met1.select($"attribute",$"set",$"roc_score" as "roc1").join(met0.select($"attribute",$"set",$"roc_score" as "roc0"), Seq("attribute","set")).withColumn("x", $"roc1"/$"roc0").groupBy("set").agg(avg($"x")).show
+----------+------------------+                                                 
|       set|            avg(x)|
+----------+------------------+
|     train|0.9888488879152876|
|validation|0.9904343646694229|
|  hyperopt|0.9924383553792211|
|      test|0.9926457172158499|
+----------+------------------+

// read in Structural zeros and previously saved data to speed this up!

val sz_sakey = spark.read.parquet(params.sz_sakey)
val sz_svkey = spark.read.load(params.sz_svkey)
val imp0ud = spark.read.option("header",true).option("inferSchema",true).csv(params.outDir + "summaries/univariate_distributions/iteration_0")
val imp1ud = spark.read.option("header",true).option("inferSchema",true).csv(params.outDir + "summaries/univariate_distributions/iteration_1")

//
val attributes_on_at_least_2_waves = sz_sakey.groupBy("sakey_1","sakey_2").agg(countDistinct($"wavekey") as "n_waves").where($"n_waves" > 1)
val sakeys_2_waves = attributes_on_at_least_2_waves.select($"sakey_1" as "sakey").distinct.union(attributes_on_at_least_2_waves.select($"sakey_2" as "sakey")).distinct.as[Int].collect.sorted
val imputed_sakeys = imp0ud.select($"sakey").distinct.as[Int].collect.sorted
val common_sakeys = sakeys_2_waves.intersect(imputed_sakeys)

val possible_queries_1 = sz_svkey.where($"sakey_1".isin(common_sakeys.toSeq: _*) and $"sakey_2".isin(common_sakeys.toSeq: _*)).select("sakey_1","svkey_1","sakey_2","svkey_2").distinct
val possible_queries_2 = imp0ud.select($"sakey",$"svkey",$"attribute" as "attribute1").join(possible_queries_1, $"sakey"===$"sakey_1" and $"svkey"===$"svkey_1").drop("sakey","svkey")
val possible_queries = imp0ud.select($"sakey",$"svkey",$"attribute" as "attribute2").join(possible_queries_2, $"sakey"===$"sakey_2" and $"svkey"===$"svkey_2").drop("sakey","svkey")
// scala> possible_queries.count
// res23: Long = 4,831,173                                                            

possible_queries.orderBy(rand()).limit(5000).write.mode("overwrite").save(params.outDir + "summaries/multivariate_distributions/randomly_chosen_audiences")

val randomly_chosen_audiences = spark.read.load(params.outDir + "summaries/multivariate_distributions/randomly_chosen_audiences")


// simulate the MV report
val jdbcUsername = "rn_survey_datascience"
val jdbcPassword = "d@Ta5ci3Nce"
val jdbcHostname = "pgsurvey01.qa.aws.resonatedigital.net"
val jdbcPort = 5432
val jdbcDatabase = "survey"
val jdbcUrl = s"jdbc:postgresql://${jdbcHostname}:${jdbcPort}/${jdbcDatabase}?user=${jdbcUsername}&password=${jdbcPassword}"
val connectionProperties = {
    val props = new java.util.Properties()
    props.setProperty("driver", "org.postgresql.Driver")
    props
}
val getAttributeValueKeysFromSurveyValueKeys = s"(select sv.key as svkey, atv.key as avkey from survey.survey_value sv, attribute.attribute_value atv where atv.survey_value_id = sv.id and sv.key in (SVKEYS) and atv.active) emp_alias_attkeys"
val getAndPairwiseNoXwave = s"(with my_attribute_value as (select key from attribute.attribute_value where key in (AVKEYS)), pairwise_combination as (select 'A' || a1.key || ' AND A' || a2.key as expression from my_attribute_value a1 cross join my_attribute_value a2 where a1 < a2) select expression, metric.*, projected_population::numeric / maximum_population::numeric distribution from pairwise_combination cross join lateral metric.audience_metric(expression, false) metric) emp_alias_pairwisedistribution_noxwave"
val getAndPairwiseWithXwave = s"(with my_attribute_value as (select key from attribute.attribute_value where key in (AVKEYS)), pairwise_combination as (select 'A' || a1.key || ' AND A' || a2.key as expression from my_attribute_value a1 cross join my_attribute_value a2 where a1 < a2) select expression, metric.*, projected_population::numeric / maximum_population::numeric distribution from pairwise_combination cross join lateral metric.audience_metric(expression, true) metric) emp_alias_pairwisedistribution_withxwave"


val focal_svkeys = randomly_chosen_audiences.select($"svkey_1" as "svkey").union(randomly_chosen_audiences.select($"svkey_2" as "svkey")).distinct.as[String].collect().map(_.toInt)
val fsvksDF = spark.read.jdbc(jdbcUrl, getAttributeValueKeysFromSurveyValueKeys.replace("SVKEYS", focal_svkeys.mkString(",")), connectionProperties)
fsvksDF.write.mode("overwrite").save(params.outDir + "summaries/multivariate_distributions/randomly_chosen_svkey_avkey")

val fsvksDF = spark.read.load(params.outDir + "summaries/multivariate_distributions/randomly_chosen_svkey_avkey")


val random_avkey_1 = fsvksDF.select($"svkey", $"avkey" as "avkey1").join(randomly_chosen_audiences, $"svkey" === $"svkey_1").drop("svkey")
val random_avkey_2 = fsvksDF.select($"svkey", $"avkey" as "avkey2").join(random_avkey_1, $"svkey" === $"svkey_2").drop("svkey")
val random_avkey_df = random_avkey_2.withColumn("audience", concat_ws(",", $"avkey1",$"avkey2")).cache
val random_audiences = random_avkey_df.select("audience").distinct.as[String].collect()


val random_audiences_donzo = random_audiences.par.map(avks => {
  println(random_audiences.indexOf(avks) / random_audiences.size.toDouble)
  // val avks = random_audiences(0)
  val aud = avks.replace(",","_")
  val andPairwiseNoXwave = spark.read.jdbc(jdbcUrl, getAndPairwiseNoXwave.replace("AVKEYS", avks), connectionProperties).withColumn("type", lit("and_noxw"))
  val andpairwiseWithXwave = spark.read.jdbc(jdbcUrl, getAndPairwiseWithXwave.replace("AVKEYS", avks), connectionProperties).withColumn("type", lit("and_xw"))
  val platform_result = andPairwiseNoXwave.union(andpairwiseWithXwave).withColumn("audience",lit(avks)).join(random_avkey_df, "audience")
  platform_result.repartition(1).write.mode("overwrite").save(params.outDir + "summaries/multivariate_distributions/randomly_chosen_platform_results/" + aud)
  true
})



val favks = fsvksDF.select("avkey").distinct.as[Int].collect().sorted
favks.size
val buckets = (0 to 100).toArray



imp0ud.select("attribute","sakey","svkey","average_imputed_distribution").join(distributions.select("attribute","sakey","svkey","prod_distribution","calc_distribution"),Seq("attribute","sakey","svkey")).withColumn("x", $"average_imputed_distribution"/$"calc_distribution").show
imp1ud.select("attribute","sakey","svkey","average_imputed_distribution").join(distributions.select("attribute","sakey","svkey","prod_distribution","calc_distribution"),Seq("attribute","sakey","svkey")).withColumn("x", $"average_imputed_distribution"/$"calc_distribution").show














imp0_univariate_distributions.withColumn("x", array(waves.toSeq.map(col(_)): _*)).withColumn("xx", sum($"x")).show