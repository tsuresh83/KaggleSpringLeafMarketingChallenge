# unif <- read.csv("/media/3TB/kag/springleafmarketingresponse/result/XGBoostUncorrelatedNumericAndFactors_2015-10-0702:07:07_suresh-le_6_submission.csv")
# eq <- read.csv("/media/3TB/kag/springleafmarketingresponse/result/XGBoostUncorrelatedNumericAndDateEqSamples_2015-10-0702:07:07_suresh-le_6_submission.csv")
rm(list=ls())
library(Hmisc)
#load("/media/3TB/kag/springleafmarketingresponse/result/XGBoostUncorrelatedNumericAndSelectFactorsEQAndUnifCombined_2015-10-0720:06:09_suresh-le_6_WS.rdata")
#m <- merge(unif,eq,by="ID")
#0.8=79303,0.7=79276
#x <- training[-which(colnames(training)%in% factorFeatures$Name)]
#x <- x[uncorrelatedCols]
#train <- cbind(splitDateFeatures,usefulFactorDummies,train)
#samples <- sample(1:nrow(training),80000)
#train <-training[numericCols$Name]
#x <- cbind(dateFeaturesSS,x)
#y<-x[-h,]
load("/media/3TB/kag/springleafmarketingresponse/result/XGBoostAllNumericAndFactorsEQThirdAndUnifCombined_2015-10-0904:52:59_suresh-le_6_WS.rdata")
y<-val
labels <- y[,c("target")]
valPredUnif <- predict(model, data.matrix(y[,-ncol(y)]))
valpredEq <- predict(modelEQ,data.matrix(y[,-ncol(y)]))
valpredThird <- predict(modelThird,data.matrix(y[,-ncol(y)]))
print(cvAUC::AUC(predictions = valPredUnif , labels = labels)) #0.7474598,0.5(non standard)
print(cvAUC::AUC(predictions = valpredEq , labels = labels))
print(cvAUC::AUC(predictions = valpredThird , labels = labels))
aucs <- numeric()
aucsAll <- numeric()
sweep <-seq(0.9,1,by=0.01)
for(i in sweep){
  #print(i)
  aucs <- c(aucs,(cvAUC::AUC(predictions = apply(cbind(valPredUnif,valpredThird),1,function(x){wtd.mean(x,c(i,1-i))}) , labels = labels)))
  #aucsAll <- c(aucsAll,(cvAUC::AUC(predictions = apply(cbind(valPredUnif,valpredEq,valpredThird),1,function(x){wtd.mean(x,c(1,1,1))}) , labels = labels)))
}
#aucsAll <-c(aucsAll,(cvAUC::AUC(predictions = apply(cbind(valPredUnif,valpredEq,valpredThird),1,function(x){wtd.mean(x,c(i,(1-i)/2,(1-i)/2))}) , labels = labels)))
aucsAll <- numeric()
aucut <- numeric()
aucue <- numeric()
aucte <- numeric()
sweepAll <- seq(0,1,by=0.1)
for(u in sweepAll){
  for(t in sweepAll){
    aucut <- c(aucut,(cvAUC::AUC(predictions = apply(cbind(valPredUnif,valpredThird),1,function(x){wtd.mean(x,c(u,(1-u)))}) , labels = labels)))
    for(e in sweepAll){
      aucue <- c(aucue,(cvAUC::AUC(predictions = apply(cbind(valPredUnif,valpredEq),1,function(x){wtd.mean(x,c(u,(1-u)))}) , labels = labels)))
      aucte <- c(aucte,(cvAUC::AUC(predictions = apply(cbind(valpredThird,valpredEq),1,function(x){wtd.mean(x,c(t,(1-t)))}) , labels = labels)))
      #aucsAll <- c(aucsAll,(cvAUC::AUC(predictions = apply(cbind(valPredUnif,valpredThird,valpredEq),1,function(x){wtd.mean(x,c(u,(1-u)/2,(1-u)/2))}) , labels = labels)))
    }
  }
}
print(paste("max auc ",max(aucs)))
print(paste("max aucAll ",max(aucsAll)))
unifWtForMaxAuc <- sweep[which(aucs==max(aucs),arr.ind = T)]
unifWtForMaxAucAll <- sweep[which(aucsAll==max(aucsAll),arr.ind = T)]
s <- data.frame(ID=submission$ID)
s$target <- apply(submission[,c(2,4)],1,function(x){wtd.mean(x,c(unifWtForMaxAuc,1-unifWtForMaxAuc))})
#predictions = m$target
#load the classifiers and run on val
#79308(0.92UnifEq)
#79299(1,1,1)
#79367(5,3,2),79348(5,2,3),79352(45,35,20)
write.csv(s,file=paste("/media/3TB/kag/springleafmarketingresponse/result/XgBoostUnifThirdMostFactors",unifWtForMaxAuc,".csv",sep=""),row.names = F)
s$target <- apply(submission[,c(2,3,4)],1,function(x){wtd.mean(x,c(1,1,1))})
write.csv(s,file=paste("/media/3TB/kag/springleafmarketingresponse/result/XgBoostUnifEqThirdMeanMostFactors","",".csv",sep=""),row.names = F)
s$target <- apply(submission[,c(2,3,4)],1,function(x){wtd.mean(x,c(0.45,0.35,0.2))})
write.csv(s,file=paste("/media/3TB/kag/springleafmarketingresponse/result/XgBoostUnifEqThird45p35p2",unifWtForMaxAuc,".csv",sep=""),row.names = F)
