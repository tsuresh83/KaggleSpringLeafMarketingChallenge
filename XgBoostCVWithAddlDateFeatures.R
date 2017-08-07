rm(list=ls())
library(doMC)
library(xgboost)
library(plyr)
library(caret)
library(pROC)
library(dummy)
set.seed(13)

startTime <- Sys.time()
equalSamples <- F
scriptName<-ifelse(equalSamples,"XgBoostEqualSamples","XGBOOSTADDLDATEFEATURES")
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

# training=training[,!(names(training) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918,"VAR_0227","VAR_0228"))]
# test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918,"VAR_0227","VAR_0228"))]
training=training[,!(names(training) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]
test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]

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
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=","),NUniqueValues=length(unique(training[,i]))))
}
training <- training[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
factorFeatures <- colTypesAndRanges[colTypesAndRanges$Type=="factor",]
#test <- test[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
training[is.na(training)] <- -1
test[is.na(test)]   <- -1
dateColsAsDate <- training[dateCols]
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
#get rid of the date columns - to be replaced with the engineered columns
#train <- training[-which(colnames(training)%in% factorFeatures$Name)]
#train <- training[-c(which(colnames(training)%in%c("VAR_0200","VAR_0404","VAR_0493","VAR_0214")),which(colnames(training)%in% dateCols))]#var_0200 has 12387 unique values - likely useless
#usefulFactors <- factorFeatures[!(factorFeatures$Name %in% c("VAR_0200","VAR_0404","VAR_0493","VAR_0214",dateCols)),]$Name
#usefulFactors <- factorFeatures[!(factorFeatures$Name %in% dateCols),]$Name
#usefulFactors <- factorFeatures[!(factorFeatures$Name %in% c("VAR_0200","VAR_0404","VAR_0493","VAR_0214",dateCols)),]$Name
usefulFactors <- factorFeatures[!(factorFeatures$Name %in% c("VAR_0200","VAR_0214",dateCols)),]$Name
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

dtrain <- xgb.DMatrix(data.matrix(train[,1:(ncol(train)-1)]), label=train$target)




dval <- xgb.DMatrix(data.matrix(val[,1:(ncol(val)-1)]), label=val$target)

watchlist <- list(eval = dval, train = dtrain)
#0.02,6,1,1,2K = eval-auc:0.787196	train-auc:0.946489
#0.02,6,0.7,0.8,2K - eval-auc:0.789811	train-auc:0.958717
#0.02,6,0.7,0.9,2K - eval-auc:0.790186	train-auc:0.959761
#0.02,6,0.7,0.9,4K - eval-auc:0.789965	train-auc:0.995402 SEEMS TO MAX OUT AT AROUND 0.7917 ~3K STEPS
#0.01,6,0.7,0.9,4K - eval-auc:0.790975	train-auc:0.960763
#0.015,6,0.7,0.9,4K - eval-auc:0.791156,0.79146 (0.79219-lb)	train-auc:0.984974

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
                    nrounds             = 4000, # changed from 300
                    verbose             = 2, 
                    #early.stop.round    = 50,
                    watchlist           = watchlist,
                    maximize            = TRUE)



modelEndTime <- Sys.time()
totalModelTime <- as.numeric((modelEndTime-modelStartTime),units="mins")
prefix <- paste(scriptName,gsub(" ","",modelEndTime),nodename,ncores,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"_WS.rdata",sep="")
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)

#engineer date columns of test
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
splitDateFeaturesTest$NValidDates <- (apply(splitDateFeaturesTest[,grepl("Y_",colnames(splitDateFeaturesTest))],1,function(x){length(which(x!=1900))}))
splitDateFeaturesTest$DateDiff <- as.numeric(dateColsAsDateTest$VAR_0204 - dateColsAsDateTest$VAR_0217,units="days")
splitDateFeaturesTest$DateDiff2 <- as.numeric(dateColsAsDateTest$VAR_0217 - dateColsAsDateTest$VAR_0075,units="days")

submission <- data.frame(ID=test$ID,target=NA)
#test1 <- test[-c(1,which(colnames(test)%in% dateCols))]
usefulFactorDummiesTest <- dummy(test[which(colnames(test) %in% usefulFactors)])
test1 <- test[-c(1,which(colnames(test)%in% factorFeatures$Name))]
test1 <- cbind(splitDateFeaturesTest,usefulFactorDummiesTest,test1)

featureNames <-colnames(train)[1:(ncol(train)-1)]
levelNotInTest <- featureNames[! (featureNames %in% colnames(test1))]
ctr <-1
for(l in levelNotInTest){
  #print(l)
  test1 <- cbind(test1,data.frame(c=as.factor(rep(0,nrow(test1)))))
}
colnames(test1) <- c(colnames(test1)[1:(ncol(test1)-length(levelNotInTest))],levelNotInTest)
test1 <- test1[featureNames]
print("test data readied")
#test1 <- test1[,2:ncol(test1)]
##end engineering


#submission$target <- NA 
target <- numeric()
print(length(featureNames))
print(ncol(test1))
print(which(colnames(test1)!=featureNames))
# for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
#   target <- c(target,as.character(predict(clf, data.matrix(test1[rows,]))))
#   
# }
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