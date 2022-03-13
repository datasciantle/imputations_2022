rm(list=ls())
require(ggplot2)

files <- list.files('~/Dropbox/RTP/guardians/imputations_2022/lifo_2105/results/v2_tuner/',full.names=T, pattern='univariate')

for(i in files){
	df <- read.csv(i)
	if(i == files[1]){x <- df}
	else{x <- rbind(x,df)}
}


x <- x[sample(1:nrow(x), nrow(x), FALSE),]

waves <- colnames(x)[grep('X', colnames(x))]
for(i in waves){
	y <- x[,c('attribute','sakey','svkey','multi','simulation_distribution',i)]
	colnames(y)[ncol(y)] <- 'wave_imputation_distribution'
	y$wave <- i
	if(i==waves[1]){
		ydf <-y
		} else {
			ydf <- rbind(ydf,y)
		}
}


ggplot(data=x, aes(x= prod_distribution, y= imputed_distribution, color= factor(iteration), shape=multi)) + geom_point(alpha=0.7) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())


ggplot(data=x, aes(x= simulation_distribution, y= imputed_distribution, color= factor(iteration), shape=multi)) + geom_point(alpha=0.3) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + geom_abline(intercept=0.05, slope=1, linetype='dashed',color='red', size=0.3) + geom_abline(intercept=-0.05, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())

y <- ydf[sample(1:nrow(ydf), size=10000, replace=F),]

ggplot(data=y, aes(x= simulation_distribution, y= wave_imputation_distribution, color= wave, shape=multi)) + geom_point(alpha=0.3,) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + geom_abline(intercept=0.05, slope=1, linetype='dashed',color='red', size=0.3) + geom_abline(intercept=-0.05, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank()) + geom_smooth(method='lm')




ggplot(data=x, aes(x= simulation_distribution, y= prod_distribution, color= factor(iteration), shape=multi)) + geom_point(alpha=0.7) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())

ggplot(data=x, aes(x= simulation_distribution, y= calc_distribution, color= factor(iteration), shape=multi)) + geom_point(alpha=0.7) + geom_abline(intercept=0, slope=1, linetype='dashed',color='red', size=0.3) + theme(panel.background=element_blank())




for(i in unique(x$iteration)){
	f <- x[x$iteration==i,]
	l2 <- (f$imputed_distribution - f$simulation_distribution)**2
	print(mean(l2), na.rm=TRUE)
	#plot(hist(l2))
	# Sys.sleep(5)
}




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


head(x[which(x$prod_distribution > 0.45 & x$prod_distribution < 0.55 & x$imputed_distribution > 0.75),])



