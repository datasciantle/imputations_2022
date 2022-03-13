///usr/bin/spark-shell --master yarn --deploy-mode client --driver-memory 12G --packages org.postgresql:postgresql:9.4-1206-jdbc42 --conf spark.maximizeResourceAllocation=true --conf spark.dynamicAllocation.enabled=true --conf spark.executor.instances=1000 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.autoBroadcastJoinThreshold=20971520 --conf spark.dynamicAllocation.maxExecutors=1000 --conf "spark.memory.storageFraction=0.1" --conf "spark.memory.fraction=0.1" //--jars "s3committer-0.5.5.jar"


val cti = spark.read.load("s3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed")
val focal_bsm = spark.read.load("s3://datasci-scantle/unified_ncs/2105/inputs/focal_bsm/*")
val dmf =  spark.read.load("s3://datasci-scantle/unified_ncs/2105/dmf_latent_distribution_predictions/self_supervised_svkey_embedding_tabnet_dmf_tf_efc216265f32_epoch_19/parquet")


val attribute_matrix_1 = focal_bsm.join(cti.select("sakey","svkey","multi","alternate_svkey_binary_single","attribute","value"), Seq("sakey","svkey","multi"))

val binomial_attributes = attribute_matrix_1.where($"multi" or (not($"svkey" === $"alternate_svkey_binary_single"))).withColumn("attribute_value", $"label")
val multinomial_attributes = attribute_matrix_1.where(not($"multi") and $"alternate_svkey_binary_single".isNull and $"label" === 1.0).withColumn("attribute_value", $"value")

val attribute_matrix_2 = binomial_attributes.union(multinomial_attributes).join(dmf.select("uid","set_uid").distinct, "uid").orderBy("sakey","svkey","uid")
val attribute_matrix_3 = attribute_matrix_2.select($"uid",$"wavekey",$"set_uid",$"sakey",$"svkey",$"multi",$"attribute", $"attribute_value" as "label", $"attribute" as "attribute2")

val n_attributes = attribute_matrix_3.select($"attribute").distinct.count

val attribute_matrix = attribute_matrix_3.repartition(n_attributes.toInt, $"attribute2").cache
attribute_matrix.count

attribute_matrix.count
attribute_matrix.write.mode("overwrite").partitionBy("attribute2").save("s3://datasci-scantle/unified_ncs/2105/inputs/attribute_matrix")

val n_respondents = attribute_matrix.select($"uid").distinct.count
val attribute_counts = attribute_matrix.groupBy("attribute").agg(countDistinct($"uid") as "n_respondents")

val complete_attribute_matrix = attribute_counts.where($"n_respondents" === n_respondents).join(attribute_matrix,"attribute").groupBy("uid","wavekey","set_uid").pivot("attribute").agg(max($"label"))
val at_least50_attribute_matrix = attribute_counts.where($"n_respondents" >= n_respondents*0.5).join(attribute_matrix,"attribute").groupBy("uid","wavekey","set_uid").pivot("attribute").agg(max($"label"))



complete_attribute_matrix.repartition(1).write.mode("overwrite").save("s3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_attribute_matrix")
at_least50_attribute_matrix.repartition(1).write.mode("overwrite").save("s3://datasci-scantle/unified_ncs/2105/lifo/inputs/at_least50_attribute_matrix")



val attribute_matrix_2 = attribute_matrix_1.withColumn("attribute_value", when($"multi", $"label").when((not($"multi") and not($"svkey"===$"alternate_svkey_binary_single")),$"label").otherwise($"value"))
val attribute_matrix_3 = attribute_matrix_2.where($"multi" or (not($"multi") and not($"svkey" === $"alternate_svkey_binary_single")) or (not($"multi") and $"alternate_svkey_binary_single".isNull and $"label" === 1.0))