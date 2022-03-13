///usr/bin/spark-shell --master yarn --deploy-mode client --driver-memory 12G --packages org.postgresql:postgresql:9.4-1206-jdbc42 --conf spark.maximizeResourceAllocation=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.instances=1000 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.autoBroadcastJoinThreshold=20971520 --conf spark.dynamicAllocation.maxExecutors=1000 --conf "spark.memory.storageFraction=0.1" --conf "spark.memory.fraction=0.1" //--jars "s3committer-0.5.5.jar"

import spark.implicits._
import org.apache.spark.sql.expressions.Window
import org.apache.hadoop.fs.{FileSystem, Path}

case class Params (
  val complete_attribute_matrix: String = "s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix",
  val dmf_latent_layer: String = "s3://datasci-scantle/unified_ncs/2105/dmf_latent_distribution_predictions/self_supervised_svkey_embedding_tabnet_dmf_tf_efc216265f32_epoch_19/parquet",
  val outDir: String = "s3://datasci-scantle/unified_ncs/experiments/2105_miceforest/iteration_0/")

val params =  new Params

val complete_attribute_matrix = spark.read.load(params.complete_attribute_matrix)
val dmf_latent_layer = spark.read.load(params.dmf_latent_layer)

val core_dmf = complete_attribute_matrix.join(dmf_latent_layer, Seq("uid","set_uid"), "inner").repartition(200).cache
core_dmf.count
core_dmf.write.mode("overwrite").option("header",true).csv(params.outDir + "inputs/core_and_dmf")

