rm(list=ls())
library(doMC)
library(xgboost)
library(plyr)
library(caret)
library(pROC)
library(dummy)
library(h2o)
library(h2oEnsemble)
library(SuperLearner)
library(cvAUC)
set.seed(13)

startTime <- Sys.time()
scriptName<-"H20ENSEMBLE"
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
trainFile <- ifelse(nodename=="bigtumor",("/home/tumor/MLExperimental/springleafmarketingresponse/data/train.rdata"),
                    ifelse(os=="Darwin",
                           ("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),
                           ("/media/3TB/kag/springleafmarketingresponse/data/train.rdata")))
load(trainFile)
training <- training[,2:ncol(training)]
print("Training data set loaded...")
testFile <- ifelse(nodename=="bigtumor",("/home/tumor/MLExperimental/springleafmarketingresponse/data/test.rdata"),
                   ifelse(os=="Darwin",
                          ("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),
                          ("/media/3TB/kag/springleafmarketingresponse/data/test.rdata")))
load(testFile)
print("Test data set loaded")
#remove irrelevant columns
training.unique.count=lapply(training, function(x) length(unique(x)))
training.unique.count_1=unlist(training.unique.count[unlist(training.unique.count)==1])
training.unique.count_2=unlist(training.unique.count[unlist(training.unique.count)==2])
training.unique.count_2=training.unique.count_2[-which(names(training.unique.count_2)=='target')]

delete_const=names(training.unique.count_1)
delete_NA56=names(which(unlist(lapply(training[,(names(training) %in% names(training.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
delete_NA89=names(which(unlist(lapply(training[,(names(training) %in% names(training.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
delete_NA918=names(which(unlist(lapply(training[,(names(training) %in% names(training.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))

#VARS to delete
#safe to remove VARS with 56, 89 and 918 NA's as they are covered by other VARS
print(length(c(delete_const,delete_NA56,delete_NA89,delete_NA918)))

training=training[,!(names(training) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918,"VAR_0227","VAR_0228"))]
test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918,"VAR_0227","VAR_0228"))]

#end removing columns
ncores <- ifelse(nodename=="bigtumor",3,
                 ifelse(os=="Darwin",2,6))
registerDoMC(ncores)
outputFolder <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/",
                       ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/result/",
                              "/media/3TB/kag/springleafmarketingresponse/result/"))



# names(training)  # 1934 variables
dateCols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169",
              "VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
colTypesAndRanges <- data.frame()
for(i in 1 : ncol(training)){
  col <- colnames(training)[i]
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=","),NUniqueValues=length(unique(training[,i])),totalNAs=length(which(is.na(training[,i])))))
}
#training <- training[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
factorFeatures <- colTypesAndRanges[colTypesAndRanges$Type=="factor",]
#test <- test[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
training[is.na(training)] <- -1
test[is.na(test)]   <- -1
dateColsAsDate <- training[dateCols]
dateTimeDF <- data.frame(apply(dateColsAsDate, 2, function(x) as.double(strptime(x, format='%d%b%y:%H:%M:%S', tz="UTC")))) #2 = columnwise
dateTimeDF[is.na(dateTimeDF)] <- -25567
colnames(dateTimeDF) <- paste(colnames(dateTimeDF),"DT",sep="")
for(col in colnames(dateColsAsDate)){
  temp <- as.vector(dateColsAsDate[col])
  dateColsAsDate[col] <- as.Date(temp[,1],"%d%b%y")
}
dateColsAsDate[is.na(dateColsAsDate)]<-as.Date("1900-01-01","%Y-%m-%d")
splitDateFeatures <- data.frame()
ctr <- 1
for(col in colnames(dateColsAsDate)){
  if(ctr==1){
    splitDateFeatures <- cbind(Y=as.integer(format(dateColsAsDate[col],"%Y" )[,1]),
                               M=as.integer(format(dateColsAsDate[col],"%m")[,1]),
                               D=as.integer(format(dateColsAsDate[col],"%d")[,1]))
  }else{
    splitDateFeatures <- cbind(splitDateFeatures,cbind(Y=as.integer(format(dateColsAsDate[col],"%Y" )[,1]),
                                                       M=as.integer(format(dateColsAsDate[col],"%m")[,1]),
                                                       D=as.integer(format(dateColsAsDate[col],"%d")[,1])))
  }
  ctr <- ctr+1
}
splitDateFeatures <- as.data.frame(splitDateFeatures)
colnames(splitDateFeatures)<- paste(colnames(splitDateFeatures),rep(1:(length(colnames(splitDateFeatures))/3),each=3),sep="_")
splitDateFeatures$NValidDates <- (apply(splitDateFeatures[,grepl("Y_",colnames(splitDateFeatures))],1,function(x){length(which(x!=1900))}))
splitDateFeatures$DateDiff <- as.numeric(dateColsAsDate$VAR_0204 - dateColsAsDate$VAR_0217,units="days")
splitDateFeatures$DateDiff2 <- as.numeric(dateColsAsDate$VAR_0217 - dateColsAsDate$VAR_0075,units="days")
splitDateFeatures <- cbind(splitDateFeatures,dateTimeDF)
splitDateFeatures$DTDiff <- splitDateFeatures$VAR_0204DT - splitDateFeatures$VAR_0217DT
splitDateFeatures$DTDiff2 <- splitDateFeatures$VAR_0217DT - splitDateFeatures$VAR_0075DT
splitDateFeatures$HOLIDAYS <- ifelse(splitDateFeatures$M_16>9 | splitDateFeatures$M_16 < 3,1,0)
weekday1 <- weekdays(dateColsAsDate$VAR_0217)
weekday2 <- weekdays(dateColsAsDate$VAR_0204)
weekdayInt1 <- as.integer(1:7)[match(weekday1,c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))]
weekdayInt2 <- as.integer(1:7)[match(weekday2,c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))]
splitDateFeatures$Weekday1 <- weekdayInt1
splitDateFeatures$Weekday2 <- weekdayInt2
#get rid of the date columns - to be replaced with the engineered columns
#train <- training[-which(colnames(training)%in% factorFeatures$Name)]
#train <- training[-c(which(colnames(training)%in%c("VAR_0200","VAR_0404","VAR_0493","VAR_0214")),which(colnames(training)%in% dateCols))]#var_0200 has 12387 unique values - likely useless
usefulFactors <- factorFeatures[!(factorFeatures$Name %in% c("VAR_0200","VAR_0404","VAR_0493","VAR_0214",dateCols)),]$Name
#usefulFactors <- factorFeatures[!(factorFeatures$Name %in% dateCols),]$Name
usefulFactorDummies <- dummy(training[which(colnames(training) %in% usefulFactors)])
print("removed problem column var_0200,var_0404,var_0493,var_0214")
train <- training[-which(colnames(training)%in% factorFeatures$Name)]
train <- cbind(splitDateFeatures,usefulFactorDummies,train)
samples <- sample(1:nrow(training),80000)
#train$target <- as.factor(train$target)
#train <- train[sample(nrow(train), 140000),]
h <- sample(nrow(train), 100000)
val<-train[-h,]
gc()
train <-train[h,]
gc()
memory <- ifelse(nodename=="bigtumor","128g",
                 ifelse(os=="Darwin","5g",
                        "50g"))
independent <- colnames(train)[1:(ncol(train)-1)]
dependent <- "target"
metalearner <- c("SL.glm")
#train$target <- as.factor(train$target)
#val$target <- as.factor(val$target)
localH2O = h2o.init(nthreads=-1,max_mem_size = memory)
print(memory)
#tumorh20 <-
# Run regression GBM on australia.hex data
trainh20 <- as.h2o(train,conn=localH2O)
valh20 <- as.h2o(val,conn=localH2O)
trainh20[,c(dependent)] <- as.factor(trainh20[,c(dependent)])
valh20[,c(dependent)] <- as.factor(valh20[,c(dependent)])

h2o.randomForest.1 <- function(..., ntrees = 1000, max_depth=8,binomial_double_trees=T) h2o.randomForest.wrapper(..., ntrees = ntrees, max_depth=max_depth,binomial_double_trees=binomial_double_trees)
h2o.randomForest.2 <- function(..., ntrees = 1000, max_depth=6,binomial_double_trees=T) h2o.randomForest.wrapper(..., ntrees = ntrees, max_depth=max_depth,binomial_double_trees=binomial_double_trees)
h2o.gbm.1 <- function(..., ntrees = 1000, max_depth=8,learn_rate=0.02) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth=max_depth,learn_rate=0.01)
h2o.gbm.2 <- function(..., ntrees = 2000, max_depth=6,learn_rate=0.01) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth=max_depth,learn_rate = 0.01)
learner <- c("h2o.randomForest.1","h2o.randomForest.2","h2o.gbm.1","h2o.gbm.2")
metalearner <- c("SL.glm")
family <- "binomial"
modelStart <- Sys.time()
fit <- h2o.ensemble(x = independent, y = dependent, 
                    training_frame = trainh20, 
                    validation_frame = valh20,
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner)#,
#cvControl = list(V = 2, shuffle = TRUE))
modelEnd <- Sys.time()
totalModelTime <- as.numeric((modelEnd-modelStart),units="mins")
print(paste("Total model time ",totalModelTime))
prefix <- paste(scriptName,gsub(" ","",startTime),nodename,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"_WS.rdata",sep="")
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
#engineer date columns of test
dateColsAsDateTest <- test[dateCols]
dateTimeDFTest <- data.frame(apply(dateColsAsDateTest, 2, function(x) as.double(strptime(x, format='%d%b%y:%H:%M:%S', tz="UTC")))) #2 = columnwise
dateTimeDFTest[is.na(dateTimeDFTest)] <- -25567
colnames(dateTimeDFTest) <- paste(colnames(dateTimeDFTest),"DT",sep="")
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
splitDateFeaturesTest$NValidDates <- (apply(splitDateFeaturesTest[,grepl("Y_",colnames(splitDateFeaturesTest))],1,function(x){length(which(x!=1900))}))
splitDateFeaturesTest$DateDiff <- as.numeric(dateColsAsDateTest$VAR_0204 - dateColsAsDateTest$VAR_0217,units="days")
splitDateFeaturesTest$DateDiff2 <- as.numeric(dateColsAsDateTest$VAR_0217 - dateColsAsDateTest$VAR_0075,units="days")
splitDateFeaturesTest <- cbind(splitDateFeaturesTest,dateTimeDFTest)
splitDateFeaturesTest$DTDiff <- splitDateFeaturesTest$VAR_0204DT - splitDateFeaturesTest$VAR_0217DT
splitDateFeaturesTest$DTDiff2 <- splitDateFeaturesTest$VAR_0217DT - splitDateFeaturesTest$VAR_0075DT
splitDateFeaturesTest$HOLIDAYS <- ifelse(splitDateFeaturesTest$M_16>9 | splitDateFeaturesTest$M_16 < 3,1,0)
weekdayTest1 <- weekdays(dateColsAsDateTest$VAR_0217)
weekdayTest2 <- weekdays(dateColsAsDateTest$VAR_0204)
weekdayTestInt1 <- as.integer(1:7)[match(weekdayTest1,c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))]
weekdayTestInt2 <- as.integer(1:7)[match(weekdayTest2,c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))]
splitDateFeaturesTest$Weekday1 <- weekdayTestInt1
splitDateFeaturesTest$Weekday2 <- weekdayTestInt2
submission <- data.frame(ID=test$ID,target=NA)
#test1 <- test[-c(1,which(colnames(test)%in% dateCols))]
usefulFactorDummiesTest <- dummy(test[which(colnames(test) %in% usefulFactors)])
test1 <- test[-c(1,which(colnames(test)%in% factorFeatures$Name))]
test1 <- cbind(splitDateFeaturesTest,usefulFactorDummiesTest,test1)

featureNames <-colnames(train)[1:(ncol(train)-1)]
levelNotInTest <- featureNames[! (featureNames %in% colnames(test1))]
ctr <-1
for(l in levelNotInTest){
  print(l)
  test1 <- cbind(test1,data.frame(c=as.factor(rep(0,nrow(test1)))))
}
colnames(test1) <- c(colnames(test1)[1:(ncol(test1)-length(levelNotInTest))],levelNotInTest)
test1 <- test1[featureNames]
print("test data readied")
test1 <- as.h2o(test1,conn=localH2O)
pred <- predict.h2o.ensemble(fit,test1)
print("target estimated")
#predictions <- (pred$pred)

submission$target <- pred$pred
endTime <- Sys.time()
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
print(paste("total time ",totalScriptTime))
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
#diagnostics....
pred1<- predict.h2o.ensemble(fit, valh20)
predictions <- as.data.frame(pred1$pred)  #third column, p1 is P(Y==1)
labels <- as.data.frame(valh20[,c("target")])[,1]


# Ensemble test AUC 
cvAUC::AUC(predictions = predictions , labels = labels)
# 0.7888723


# Base learner test AUC (for comparison)
L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred1$basepred)[,l], labels = labels)) 
data.frame(learner, auc)