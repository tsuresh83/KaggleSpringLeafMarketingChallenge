rm(list=ls())
library(e1071)
library(caret)
library(RANN)
library(pROC)
library(doMC)
# did not improve over EqualSamples69P.R
# training <- read.csv("../input/train.csv")
# test  <- read_csv("../input/test.csv")
#test <-read.csv("/media/3TB/kag/springleafmarketingresponse/data/test.csv")
startTime <- Sys.time()
scriptName <-"AdaBagAfterRFE"
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/train.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata")))
print("Training data set loaded...")
ncores <- ifelse(nodename=="bigtumor",15,
                 ifelse(os=="Darwin",2,6))
registerDoMC(ncores)
outputFolder <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/",
                       ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/result/",
                              "/media/3TB/kag/springleafmarketingresponse/result/"))
optimizedVars <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/rfeOptimizedVariables.csv",
                        ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/result/rfeOptimizedVariables.csv",
                               "/media/3TB/kag/springleafmarketingresponse/result/rfeOptimizedVariables.csv"))
optimizedVars <- as.character(read.csv(optimizedVars)[,1])
dir.create(outputFolder,recursive=T)
training[(is.na(training))]<- (-90909090)
#determing types and ranges of columns
colTypesAndRanges <- data.frame()
for(i in 1 : ncol(training)){
  col <- colnames(training)[i]
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=","),NUniqueValues=length(unique(training[,i]))))
}
colTypesAndRanges$NCHAR <- nchar(as.character(colTypesAndRanges$Values))
charCols <- colTypesAndRanges[colTypesAndRanges$Type=="character",]
numericCols <- colTypesAndRanges[colTypesAndRanges$Type %in% c("numeric","integer") ,]
numericCols <- numericCols[numericCols$NUniqueValues > 1,] # discard features with just 1 unique value
dateCols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169",
              "VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
factorCols <- colTypesAndRanges[colTypesAndRanges$Type=="factor",]
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
nonDateFactors <- training[setdiff(factorCols[factorCols$NUniqueValues>2,]$Name,dateCols)]
for(i in 1:length(colnames(nonDateFactors))){
  nonDateFactors[,i] <- as.integer(nonDateFactors[,i])
}
stateCols <- c("VAR_0274","VAR_0237")
designationCols <- c("VAR_0493","VAR_0404") # seem like the designations of the potential customer
dataSource <- c("VAR_1934")
logicalCols <- c("VAR_0236","VAR_0232","VAR_0230","VAR_0226")
#logical is useless - always 'false' or 'NA'
dateOtherFactorsNumericTraining <- cbind(splitDateFeatures,nonDateFactors,training[numericCols$Name])
#get rid of non-date factors - they don't have high ranks and the integer value of these factors can influence fitting and prediction
dateOtherFactorsNumericTraining <- dateOtherFactorsNumericTraining[setdiff(optimizedVars,c("VAR_0200","VAR_0274","VAR_0237"))]
print(paste("Total number of features ",ncol(dateOtherFactorsNumericTraining)))
dateOtherFactorsNumericTraining$target <- as.factor(training$target)# target needs to be added separately
target0 <- dateOtherFactorsNumericTraining[dateOtherFactorsNumericTraining$target==0,]
target1 <- dateOtherFactorsNumericTraining[dateOtherFactorsNumericTraining$target==1,]
sampleTraining <- rbind(target0[sample(1:nrow(target0),10000,replace = T),],
                        target1[sample(1:nrow(target1),10000,replace = T),])
sampleTraining$target <- as.character(sampleTraining$target)
sampleTraining[sampleTraining$target=='0',]$target <- 'X0'
sampleTraining[sampleTraining$target=='1',]$target <- 'X1'
sampleTraining$target <- as.factor(sampleTraining$target)
#control <- trainControl(classProbs = T,summaryFunction = twoClassSummary)
control <- trainControl(classProbs = T,summaryFunction = twoClassSummary)
# run the RFE algorithm
modelStart <- Sys.time()
print("model starting...")
#best ROC 0.8046867 with mtry 199
#model <- train(target~., data=sampleTraining, method="rf", trControl=control,metric = "ROC")
model <- train(target~., data=sampleTraining, method="AdaBag", trControl=control,metric = "ROC",tuneGrid=expand.grid(mfinal=seq(100,396,50),maxdepth=seq(25,30,1)))
modelEnd <- Sys.time()
prefix <- paste(scriptName,gsub(" ","",modelEnd),nodename,ncores,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"UptoModel_WS.rdata",sep="")
modelingTime <- as.numeric((modelEnd-modelStart),units="mins")

save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
######################
#prediction
######################
predictionStart <- Sys.time()
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/test.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/test.rdata")))
#####modify test date columns
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

#######
#trim test columns
test[(is.na(test))]<- (-90909090)
test1 <- cbind(splitDateFeaturesTest,test)
#test1 <- test1[setdiff(predictors(model),c("VAR_0200","VAR_0274","VAR_0237"))]
#######
submission <- data.frame(ID=test$ID)
submission$target <- NA 
target <- character()
for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
  #submission[rows, "target"] <- predict(model, test1[rows,])
  target <- c(target,as.character(predict(model, test1[rows,])))
}
submission$target <- target
submission[submission$target=="X0",]$target <-0
submission[submission$target=="X1",]$target <-1
totalPredictionTime <- Sys.time() - predictionStart
endTime <- Sys.time()
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
print(paste("total time ",totalScriptTime))
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
workSpaceFileName <- paste(outputFolder,prefix,"_UptoPredictionWS.rdata",sep="")
write.csv(submission,submissionFileName,row.names = F)
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)



