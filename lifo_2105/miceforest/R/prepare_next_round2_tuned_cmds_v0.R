rm(list=ls())

# completed <- system(paste0('aws s3 ls s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/completeness/'),intern=TRUE)
# completed <- as.vector(sapply(completed, function(x) tail(strsplit(x,' ')[[1]],1)))
# focal_models <- unique(completed)

model_plan <- read.csv('/Users/daniel.scantlebury/Dropbox/RTP/guardians/imputations_2022/lifo_2105/model_plan/simple_model_plan')
focal_models <- model_plan[which(model_plan$proposed_round == 0 & model_plan$needs_imputation=='True'),]
focal_models <- focal_models$attribute[-which(focal_models$attribute %in% model_plan$attribute[which(model_plan$proposed_round == 0 & model_plan$needs_imputation=='False')])]
focal_models <- unique(focal_models)

base_s3_path <- 's3://datasci-scantle/unified_ncs/experiments/2105_miceforest'
model_version <- 'lifo_2105'
s3_attribute_matrix <- 's3://datasci-scantle/unified_ncs/2105/inputs/attribute_matrix'
s3_complete_taxonomy_file <- 's3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed'
s3_distributions <- 's3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions'
s3_feature_input_to_round <- 's3://datasci-scantle/unified_ncs/experiments/2105_miceforest/iteration_0/inputs/core_and_dmf'
s3_valid_respondents <- 's3://datasci-scantle/unified_ncs/2105/bsm/data/validRespondents'

iteration <- strsplit(s3_feature_input_to_round,'iteration_')[[1]][2]
iteration <- strsplit(iteration,'/')[[1]][1]



cmd_base <- 'sudo docker run --net=host --rm -e SEED_PATH=s3://datasci-scantle/seed/docker/ -v /tmp:/tmp 126345656468.dkr.ecr.us-east-1.amazonaws.com/quantumplethodon:rkumar-tf-001 python3 /tmp/docker/lgbm_imputation_round2_v0.py --base_s3_path this_base_s3_path --model_version this_model_version --s3_attribute_matrix this_s3_attribute_matrix --s3_complete_taxonomy_file this_s3_complete_taxonomy_file --s3_distributions this_s3_distributions --s3_feature_input_to_round this_s3_feature_input_to_round --s3_valid_respondents this_s3_valid_respondents --model this_model --hyperopt_max_evals 20 --redo False'
cmd_base <- gsub('this_base_s3_path', base_s3_path, cmd_base)
cmd_base <- gsub('this_model_version', model_version, cmd_base)
cmd_base <- gsub('this_s3_attribute_matrix', s3_attribute_matrix, cmd_base)
cmd_base <- gsub('this_s3_complete_taxonomy_file', s3_complete_taxonomy_file, cmd_base)
cmd_base <- gsub('this_s3_distributions', s3_distributions, cmd_base)
cmd_base <- gsub('this_s3_feature_input_to_round', s3_feature_input_to_round, cmd_base)
cmd_base <- gsub('this_s3_valid_respondents', s3_valid_respondents, cmd_base)



cmds <- {}
for(i in focal_models){
    cmd <- gsub('this_model', i, cmd_base)
    cmds <- rbind(cmds,cmd)
}

cat(cmds[2])

cmds2 <- cbind(focal_models,cmds)
colnames(cmds2) <- c('model','cmd')

cmd_file1 <- paste0('~/get_boosted_with_dmf_iteration',iteration,'.txt')
cmd_file2 <- paste0('~/expected_models_iteration',iteration,'.txt')

write.table(cmds, file = cmd_file1,  row.names = F, col.names = F, quote = F)
write.csv(cmds2, file = cmd_file2,  row.names = F)

system(paste0('aws s3 cp ', cmd_file1, ' ', base_s3_path, '/iteration_', iteration, '/focal_loss/expected_models/'),intern=TRUE)
system(paste0('aws s3 cp ', cmd_file2, ' ', base_s3_path, '/iteration_', iteration, '/focal_loss/expected_models/'),intern=TRUE)


AWS_DEFAULT_REGION=us-east-1 aws sqs purge-queue --queue-url https://sqs.us-east-1.amazonaws.com/126345656468/sqs-daniel-scantlebury
AWS_DEFAULT_REGION=us-east-1  ./quantumPlethodon/aws_runner/submit.py -q sqs-daniel-scantlebury get_boosted_with_dmf_iteration.txt
AWS_DEFAULT_REGION=us-east-1 aws autoscaling update-auto-scaling-group --auto-scaling-group-name quantumPlethodon-daniel-scantlebury --desired-capacity=50 --min-size=50 --max-size=50
AWS_DEFAULT_REGION=us-east-1 aws autoscaling update-auto-scaling-group --auto-scaling-group-name quantumPlethodon-daniel-scantlebury --desired-capacity=650 --min-size=650 --max-size=650



completed <- system(paste0('aws s3 ls ' , base_s3_path, '/iteration_',iteration,'/focal_loss/completeness/'),intern=TRUE)
completed <- as.vector(sapply(completed, function(x) tail(strsplit(x,' ')[[1]],1)))

mia <- focal_models[-which(focal_models%in%completed)]
mia





