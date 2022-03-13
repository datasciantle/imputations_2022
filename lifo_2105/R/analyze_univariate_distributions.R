rm(list=ls())
require(ggplot2)

it0 <- read.csv('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/results/univariate_iteration_0.csv')
it1 <- read.csv('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/results/univariate_iteration_1.csv')

dists <- read.csv('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/model_plan/simple_model_plan')

it0$att <- paste(it0$sakey,it0$svkey,sep='_')
it1$att <- paste(it1$sakey,it1$svkey,sep='_')
dists$att <- paste(dists$sakey, dists $svkey,sep='_')

m <- match(it0$att, it1$att)
it0$round1 <- it1$average_imputed_distribution[m]

m <- match(it0$att, dists$att)
it0$prod_distribution <- dists$prod_distribution[m]
it0$calc_distribution <- dists$calc_distribution[m]
it0$multi <- dists$multi[m]

ggplot(data=it0, aes(x= average_imputed_distribution, y= round1, color=multi)) + geom_point(alpha=0.3) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.1) + theme(panel.background=element_blank())


ggplot(data=it0, aes(x= prod_distribution, y= average_imputed_distribution, color=multi)) + geom_point(alpha=0.3) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.1) + theme(panel.background=element_blank())


ggplot(data=it0, aes(x= prod_distribution, y= round1, color=multi)) + geom_point(alpha=0.3) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.1) + theme(panel.background=element_blank())

x0 <- it0[,c('sakey','svkey','multi','average_imputed_distribution','prod_distribution')]
colnames(x0)[4] <- 'imputed_distribution'
x0$round <- '0'
x1 <- it0[,c('sakey','svkey','multi','round1','prod_distribution')]
colnames(x1)[4] <- 'imputed_distribution'
x1 $round <- '1'
x <- rbind(x0, x1)


ggplot(data=x, aes(x= prod_distribution, y= imputed_distribution, color=round, shape='multi')) + geom_point(alpha=0.3) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())

mean(x0$imputed_distribution - x0$prod_distribution, na.rm=T)
mean(x1$imputed_distribution - x1$prod_distribution, na.rm=T)
> mean(x0$imputed_distribution - x0$prod_distribution, na.rm=T)
[1] 0.03783302
> mean(x1$imputed_distribution - x1$prod_distribution, na.rm=T)
[1] 0.03391829







