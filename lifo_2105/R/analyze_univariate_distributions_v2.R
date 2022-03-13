rm(list=ls())
require(ggplot2)

files <- list.files('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/results/v2_tuner/',full.names=T, pattern='univariate')
dists <- read.csv('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/model_plan/simple_model_plan')

for(i in files){
	iteration <- strsplit(i,'iteration_')[[1]][[2]]
	iteration <- gsub('.csv','',iteration)
	df <- read.csv(i)
	df$iteration <- iteration
	if(i == files[1]){x <- df}
	else{x <- rbind(x,df)}
}

x$att <- paste(x$sakey,x$svkey,sep='_')
dists$att <- paste(dists$sakey, dists$svkey,sep='_')

m <- match(x$att, dists$att)
x$prod_distribution <- dists$prod_distribution[m]
x$calc_distribution <- dists$calc_distribution[m]
x$multi <- dists$multi[m]


x <- x[sample(1:nrow(x), nrow(x), FALSE),]

ggplot(data=x, aes(x= prod_distribution, y= average_imputed_distribution, color=iteration, shape=multi)) + geom_point(alpha=0.7) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())

ggplot(data=x[x$iteration==2,], aes(x= prod_distribution, y= average_imputed_distribution, color=iteration, shape=multi)) + geom_point(alpha=0.7) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())



ggplot(data=x, aes(x= calc_distribution, y= average_imputed_distribution, color=iteration, shape='multi')) + geom_point(alpha=0.3) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())

for(i in unique(x$iteration)){
	f <- x[x$iteration==i,]
	l2 <- (f$calc_distribution - f$average_imputed_distribution)**2
	print(mean(l2), na.rm=TRUE)
	plot(hist(l2))
	Sys.sleep(5)
}


mean(x0$imputed_distribution - x0$prod_distribution, na.rm=T)
mean(x1$imputed_distribution - x1$prod_distribution, na.rm=T)
> mean(x0$imputed_distribution - x0$prod_distribution, na.rm=T)
[1] 0.03783302
> mean(x1$imputed_distribution - x1$prod_distribution, na.rm=T)
[1] 0.03391829







