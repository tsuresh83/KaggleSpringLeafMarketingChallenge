all.models <- list(xgbModel,adaBag)
names(all.models) <- sapply(all.models, function(x) x$method)
print(sort(sapply(all.models, function(x) max(x$results$ROC))))
print(modelCor(resamples(all.models)))
all.modelsbkup <- all.models
class(all.models) <-'caretList'
save(list=ls(all.names=T),file= paste(outputFolder,prefix,"_WS.rdata",sep=""),envir = .GlobalEnv)
library(data.table)
source( "/media/3TB/kag/caretEnsemble-master/R/helper_functions.R")
assignInNamespace("makePredObsMatrix",makePredObsMatrix, ns="caretEnsemble")
greedyStart <- Sys.time()
greedy <- caretEnsemble(all.models)
glm_ensemble <- caretStack(
  all.models, 
  method='glm',
  metric='ROC',
  trControl=trainControl(
    method='boot',
    number=20,
    savePredictions=TRUE,
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)
ensembleEnd <- Sys.time()
print(paste("total stacking time ",as.numeric((ensembleEnd-greedyStart),units="mins")))


dateColsAsDateTest <- test[dateCols]
for(col in colnames(dateColsAsDateTest)){
  temp <- as.vector(dateColsAsDateTest[col])
  dateColsAsDateTest[col] <- as.Date(temp[,1],"%d%b%y")
}
dateColsAsDateTest[is.na(dateColsAsDateTest)]<-as.Date("1900-01-01","%Y-%m-%d")
splitDateFeaturesTest <- data.frame()
ctr <- 1
for(col in colnames(dateColsAsDateTest)){
  if(ctr==1){
    splitDateFeaturesTest <- cbind(Y=as.integer(format(dateColsAsDateTest[col],"%Y" )[,1]),
                                   M=as.integer(format(dateColsAsDateTest[col],"%m")[,1]),
                                   D=as.integer(format(dateColsAsDateTest[col],"%d")[,1]))
  }else{
    splitDateFeaturesTest <- cbind(splitDateFeaturesTest,cbind(Y=as.integer(format(dateColsAsDateTest[col],"%Y" )[,1]),
                                                               M=as.integer(format(dateColsAsDateTest[col],"%m")[,1]),
                                                               D=as.integer(format(dateColsAsDateTest[col],"%d")[,1])))
  }
  ctr <- ctr+1
}
splitDateFeaturesTest <- as.data.frame(splitDateFeaturesTest)
colnames(splitDateFeaturesTest)<- paste(colnames(splitDateFeaturesTest),rep(1:(length(colnames(splitDateFeaturesTest))/3),each=3),sep="_")
submission <- data.frame(ID=test$ID,target=NA)
test1 <- test[,2:ncol(test)]
#test1 <- test1[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
#test1 <- test1[-c(which(colnames(test1)%in% factorFeatures$Name))]
#test1 <- cbind(splitDateFeaturesTest,test1)
featureNames <-colnames(train)[1:(ncol(train)-1)]
if(T){
  usefulFactorDummiesTest <- dummy(test[which(colnames(test) %in% usefulFactors)])
  test1 <- test[-c(1,which(colnames(test)%in% factorFeatures$Name))]
  test1 <- cbind(splitDateFeaturesTest,usefulFactorDummiesTest,test1)
  levelNotInTest <- featureNames[! (featureNames %in% colnames(test1))]
  ctr <-1
  for(l in levelNotInTest){
    print(l)
    test1 <- cbind(test1,data.frame(c=as.factor(rep(0,nrow(test1)))))
  }
  colnames(test1) <- c(colnames(test1)[1:(ncol(test1)-length(levelNotInTest))],levelNotInTest)
  
}else{
  test1 <- cbind(splitDateFeaturesTest,test1)
}

test1 <- test1[featureNames]
##end engineering
for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
  submission[rows, "target"] <- predict(glm_ensemble, newdata=test1[rows,],type="prob")
  
}
# submission[submission$target==1,]$target <- 0
# submission[submission$target==2,]$target <- 1
prefix<-paste("CaretStackEnsembleXgAdaBoostProb",gsub(" ","",startTime),nodename,ncores,sep="_")
submissionFileName <- paste("/media/3TB/kag/springleafmarketingresponse/result/",prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)