rm(list=ls())


completed <- system(paste0('aws s3 ls s3://datasci-scantle/unified_ncs/experiments/2105/iteration_0/completeness/'),intern=TRUE)
completed <- as.vector(sapply(completed, function(x) tail(strsplit(x,' ')[[1]],1)))

base_s3_path <- 's3://datasci-scantle/unified_ncs/experiments/2105'
model_version <- 'lifo_2105'
s3_attribute_matrix <- 's3://datasci-scantle/unified_ncs/2105/inputs/attribute_matrix'
s3_distributions <- 's3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions'
s3_previous_imputation <- 's3://datasci-scantle/unified_ncs/experiments/2105/iteration_3/inputs/imputed_results_from_previous_round_csv'

iteration <- strsplit(s3_previous_imputation,'iteration_')[[1]][2]
iteration <- strsplit(iteration,'/')[[1]][1]

focal_models <- unique(completed)


cmd_base <- 'sudo docker run --net=host --rm -e SEED_PATH=s3://datasci-scantle/seed/docker/ -v /tmp:/tmp 126345656468.dkr.ecr.us-east-1.amazonaws.com/quantumplethodon:rkumar-tf-001 python3 /tmp/docker/lgbm_dmf_imputer_focal_loss_nextIteration_v1.py --base_s3_path this_base_s3_path --model_version this_model_version --s3_attribute_matrix this_s3_attribute_matrix --s3_distributions this_s3_distributions --s3_previous_imputation this_s3_previous_imputation --model this_model --iteration this_iteration'
cmd_base <- gsub('this_base_s3_path', base_s3_path, cmd_base)
cmd_base <- gsub('this_model_version', model_version, cmd_base)
cmd_base <- gsub('this_s3_attribute_matrix', s3_attribute_matrix, cmd_base)
cmd_base <- gsub('this_s3_distributions', s3_distributions, cmd_base)
cmd_base <- gsub('this_s3_previous_imputation', s3_previous_imputation, cmd_base)
cmd_base <- gsub('this_iteration', iteration, cmd_base)




cmds <- {}
for(i in focal_models){
    cmd <- gsub('this_model', i, cmd_base)
    cmds <- rbind(cmds,cmd)
}

cat(cmds[2])

cmds2 <- cbind(focal_models,cmds)
colnames(cmds2) <- c('model','cmd')

cmd_file1 <- paste0('~/get_boosted_with_dmf_iteration',iteration,'.txt')
cmd_file2 <- paste0('~/xpected_models_iteration',iteration,'.txt')

write.table(cmds, file = cmd_file1,  row.names = F, col.names = F, quote = F)
write.csv(cmds2, file = cmd_file2,  row.names = F)

system(paste0('aws s3 cp ', cmd_file1, ' ', base_s3_path, '/iteration_', iteration, '/expected_models/'),intern=TRUE)
system(paste0('aws s3 cp ', cmd_file2, ' ', base_s3_path, '/iteration_', iteration, '/expected_models/'),intern=TRUE)


AWS_DEFAULT_REGION=us-east-1 aws sqs purge-queue --queue-url https://sqs.us-east-1.amazonaws.com/126345656468/sqs-daniel-scantlebury
AWS_DEFAULT_REGION=us-east-1  ./quantumPlethodon/aws_runner/submit.py -q sqs-daniel-scantlebury get_boosted_next_iteration.txt
AWS_DEFAULT_REGION=us-east-1 aws autoscaling update-auto-scaling-group --auto-scaling-group-name quantumPlethodon-daniel-scantlebury --desired-capacity=50 --min-size=50 --max-size=50
AWS_DEFAULT_REGION=us-east-1 aws autoscaling update-auto-scaling-group --auto-scaling-group-name quantumPlethodon-daniel-scantlebury --desired-capacity=650 --min-size=650 --max-size=650



completed <- system(paste0('aws s3 ls s3://datasci-scantle/unified_ncs/experiments/2105/iteration_',iteration,'/completeness/'),intern=TRUE)
completed <- as.vector(sapply(completed, function(x) tail(strsplit(x,' ')[[1]],1)))

mia <- focal_models[-which(focal_models%in%completed)]
# cmds <- {}
# for(i in mia){
#     cmd <- gsub('this_model', i, cmd_base)
#     cmds <- rbind(cmds,cmd)
# }



# write.table(cmds, file = '~/get_boosted_with_dmf_iterationNext.txt',  row.names = F, col.names = F, quote = F)

