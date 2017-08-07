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
scriptName<-"XGBoostUncorrelatedNumericAndDateEqSamples"
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
dateFeaturesSS <- splitDateFeatures[c("Y_2","M_2","D_2","Y_15","M_15","D_15",
                                      "Y_16","M_16","D_16","VAR_0217DT","VAR_0204DT",
                                      "VAR_0075DT","NValidDates","DateDiff","DateDiff2",
                                      "Weekday1","Weekday2","HOLIDAYS")]

#find uncorrelated numeric columns
numericCols <- colTypesAndRanges[colTypesAndRanges$Type %in% c("numeric","integer") ,]
# numericColsTraining <- training[numericCols$Name]
# numericColsWOID <- numericColsTraining[-c(1)]
# numericSample <- numericColsWOID[sample(1:nrow(numericColsWOID),size = 0.2*nrow(numericColsWOID),replace = T),]
# numericCorrInSample <- cor(numericSample[-ncol(numericSample)])
# highlyCorrelatedInSample <- findCorrelation(numericCorrInSample, cutoff=0.7)
#uncorrelatedCols <- colnames(numericSample[-highlyCorrelatedInSample])
#end uncorrelated numeric columns
#print("uncorrelated numeric variables found")
#usefulFactors <- factorFeatures[!(factorFeatures$Name %in% c("VAR_0200","VAR_0404","VAR_0493","VAR_0214",dateCols)),]$Name
#usefulFactorDummies <- dummy(training[which(colnames(training) %in% usefulFactors)])

#train <- training[-which(colnames(training)%in% factorFeatures$Name)]
#train <- train[uncorrelatedCols]
#train <- cbind(splitDateFeatures,usefulFactorDummies,train)
#samples <- sample(1:nrow(training),80000)
train <-training[numericCols$Name]
train <- cbind(dateFeaturesSS,train)
h <- sample(nrow(train), 100000)
val<-train[-h,]
gc()
train <-train[h,]
gc()
train0s <- train[train$target==0,]
train1s <- train[train$target==1,]
train0s <- train[1:nrow(train1s),]
eqSamples <- rbind(train1s,train0s)
eqSamples <- eqSamples[ sample(1:nrow(eqSamples)),] #shuffle rows - just to be sure...
#dtrain <- xgb.DMatrix(data.matrix(train[,1:(ncol(train)-1)]), label=train$target)
dtrain <- xgb.DMatrix(data.matrix(eqSamples[,1:(ncol(eqSamples)-1)]), label=eqSamples$target)



dval <- xgb.DMatrix(data.matrix(val[,1:(ncol(val)-1)]), label=val$target)

watchlist <- list(eval = dval, train = dtrain)
#0.015,6,0.7,0.9,4K <- eval-auc:0.790956	train-auc:0.985359
#0.015,6,0.7,0.9,4K - eval-auc:0.791084	train-auc:0.985444 - HOLIDAY / WEEKDAY FEATURES
param <- list(  objective           = "binary:logistic", 
                #booster = "gblinear",
                #gamma               =10,
                eta                 = 0.015,
                max_depth           = 6,  
                subsample           = 0.7,
                colsample_bytree    = 0.9,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
)
modelStartTime <- Sys.time()
# model <- xgb.cv(params = param, data=xgb.DMatrix(data.matrix(train[,1:(ncol(train)-1)]), label=train$target), nfold=5, label = NULL,
#                 missing = NULL, prediction = TRUE, showsd = TRUE, metrics = list("auc"),
#                 obj = "binary:logistic", feval = NULL, stratified = F, folds = NULL,
#                 verbose = T, print.every.n = 1L, early.stop.round = NULL,
#                 maximize = TRUE)

model <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 3000, # changed from 300
                      verbose             = 2, 
                      #early.stop.round    = 50,
                      watchlist           = watchlist,
                      maximize            = TRUE)



modelEndTime <- Sys.time()
totalModelTime <- as.numeric((modelEndTime-modelStartTime),units="mins")
prefix <- paste(scriptName,gsub(" ","",modelEndTime),nodename,ncores,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"_WS.rdata",sep="")
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
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
dateFeaturesTestSS <- splitDateFeaturesTest[c("Y_2","M_2","D_2","Y_15","M_15","D_15",
                                              "Y_16","M_16","D_16","VAR_0217DT","VAR_0204DT",
                                              "VAR_0075DT","NValidDates","DateDiff","DateDiff2",
                                              "Weekday1","Weekday2","HOLIDAYS")]
submission <- data.frame(ID=test$ID,target=NA)
#test1 <- test[-c(1,which(colnames(test)%in% dateCols))]
#usefulFactorDummiesTest <- dummy(test[which(colnames(test) %in% usefulFactors)])
test1 <- test[-c(1,which(colnames(test)%in% factorFeatures$Name))]
test1 <- cbind(dateFeaturesTestSS,test1)
#test1 <- test1[uncorrelatedCols[1:(length(uncorrelatedCols)-1)]]
#test1 <- cbind(splitDateFeaturesTest,usefulFactorDummiesTest,test1)

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
target <- numeric()
print(length(featureNames))
print(ncol(test1))
print(which(colnames(test1)!=featureNames))
for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
  target <- c(target,(predict(model, data.matrix(test1[rows,]))))
  
}
print("target estimated")
submission$target <- target
# submission[submission$target==1,]$target <-0
# submission[submission$target==2,]$target <-1
endTime <- Sys.time()
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
print(paste("total time ",totalScriptTime))
prefix <- paste(scriptName,gsub(" ","",endTime),nodename,ncores,sep="_")
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)
#786 with eq, 792 with uniform, 791 with mean - explore factor variables, remove correlated numeric variables.