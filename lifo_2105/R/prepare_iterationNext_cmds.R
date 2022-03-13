rm(list=ls())


model_plan <- read.csv('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/model_plan/simple_model_plan')
base_s3_path <- 's3://datasci-scantle/unified_ncs/experiments/2105'
model_version <- 'lifo_2105'
s3_pickle_file <- 's3://datasci-scantle/unified_ncs/experiments/2105/python_inputs/iteration_0.pickle'

focal_models <- unique(model_plan$attribute[which(model_plan$proposed_round==0 & model_plan$needs_imputation=='True')])


model = '310152_310154'


cmd_base <- 'sudo docker run --net=host --rm -e SEED_PATH=s3://datasci-scantle/seed/docker/ -v /tmp:/tmp 126345656468.dkr.ecr.us-east-1.amazonaws.com/quantumplethodon:rkumar-tf-001 python3 /tmp/docker/lgbm_dmf_imputer_focal_loss_iteration0_v0.py --base_s3_path this_base_s3_path --model_version this_model_version --s3_pickle_file this_s3_pickle_file --model this_model --iteration 0'


cmd_base <- gsub('this_base_s3_path', base_s3_path, cmd_base)
cmd_base <- gsub('this_model_version', model_version, cmd_base)
cmd_base <- gsub('this_s3_pickle_file', s3_pickle_file, cmd_base)




cmds <- {}
for(i in focal_models){
    cmd <- gsub('this_model', i, cmd_base)
    cmds <- rbind(cmds,cmd)
}


cmds2 <- cbind(focal_models,cmds)
colnames(cmds2) <- c('model','cmd')

write.table(cmds, file = '~/get_boosted_with_dmf_iterationNext.txt',  row.names = F, col.names = F, quote = F)
write.csv(cmds2, file = '~/expected_models_iterationNext.txt',  row.names = F)

system(paste0('aws s3 cp ~/get_boosted_with_dmf_iterationNext.txt s3://datasci-scantle/unified_ncs/experiments/2105/iteration_1/expected_models/'),intern=TRUE)
system(paste0('aws s3 cp ~/expected_models_iterationNext.txt s3://datasci-scantle/unified_ncs/experiments/2105/iteration_1/expected_models/'),intern=TRUE)


AWS_DEFAULT_REGION=us-east-1 aws sqs purge-queue --queue-url https://sqs.us-east-1.amazonaws.com/126345656468/sqs-daniel-scantlebury
AWS_DEFAULT_REGION=us-east-1  ./quantumPlethodon/aws_runner/submit.py -q sqs-daniel-scantlebury get_boosted_with_dmf_iterationNext.txt
AWS_DEFAULT_REGION=us-east-1 aws autoscaling update-auto-scaling-group --auto-scaling-group-name quantumPlethodon-daniel-scantlebury --desired-capacity=60 --min-size=60 --max-size=60
AWS_DEFAULT_REGION=us-east-1 aws autoscaling update-auto-scaling-group --auto-scaling-group-name quantumPlethodon-daniel-scantlebury --desired-capacity=300 --min-size=300 --max-size=300



completed <- system(paste0('aws s3 ls s3://datasci-scantle/unified_ncs/experiments/2105/iteration_1/completeness/'),intern=TRUE)
completed <- as.vector(sapply(completed, function(x) tail(strsplit(x,' ')[[1]],1)))

mia <- focal_models[-which(focal_models%in%completed)]
mia <- model_plan[which(model_plan$attribute%in%mia),]
plot(mia$n_respondents ~ mia$calc_distribution, pch=19, cex=0.3, col=c('blue','black')[(mia$multi=='True') + 1])

