rm(list=ls())
library(e1071)
library(caret)
library(RANN)
library(pROC)
library(doMC)

# training <- read.csv("../input/train.csv")
# test  <- read_csv("../input/test.csv")
#test <-read.csv("/media/3TB/kag/springleafmarketingresponse/data/test.csv")
os <- Sys.info()[["sysname"]]
ifelse(os=="Darwin",load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata"))
ifelse(os=="Darwin",load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),load("/media/3TB/kag/springleafmarketingresponse/data/test.rdata"))
nCores <- ifelse(os=="Darwin",2,6)
registerDoMC(nCores)
training[(is.na(training))]<- (-90909090)
#test[(is.na(test))]<- (-90909090)
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
splitDateFeatures <- data.frame()
ctr <- 1
for(col in colnames(dateColsAsDate)){
  temp <- as.vector(dateColsAsDate[col])
  dateColsAsDate[col] <- as.Date(temp[,1],"%d%b%y")
}
dateColsAsDate[is.na(dateColsAsDate)]<-as.Date("1900-01-01","%Y-%m-%d")
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

stateCols <- c("VAR_0274","VAR_0237")
designationCols <- c("VAR_0493","VAR_0404") # seem like the designations of the potential customer
dataSource <- c("VAR_1934")
logicalCols <- c("VAR_0236","VAR_0232","VAR_0230","VAR_0226")
#logical is useless - always 'false' or 'NA'
#numericColsTraining <- training[numericCols$Name]
factorColsTraining <- training[setdiff(factorCols[factorCols$NUniqueValues>2,]$Name,
                                       dateCols)]
factorColsTraining <- cbind(factorColsTraining,splitDateFeatures)
factorColsTraining$target <- training$target
# numericColsTrainingWOIDAndTarget <- numericColsTraining[-c(1,ncol(numericColsTraining))]
# numericColsWOID <- numericColsTraining[-c(1)]
# numericColsWOID$target <- training$target
# numericColsWOID[numericColsWOID$target==0,]$target <-"X0"
# numericColsWOID[numericColsWOID$target==1,]$target <-"X1"
factorColsTraining[factorColsTraining$target==0,]$target <-"X0"
factorColsTraining[factorColsTraining$target==1,]$target <-"X1"
#numericColsWOID$target <- as.factor(numericColsWOID$target)
#X0Numeric <- numericColsWOID[numericColsWOID$target=="X0",]
#X1Numeric <- numericColsWOID[numericColsWOID$target=="X1",]
X0Factor <- factorColsTraining[factorColsTraining$target =="X0",]
X1Factor <- factorColsTraining[factorColsTraining$target=="X1",]
factorColsTraining$target <- as.factor(factorColsTraining$target)
x0Rows <- sample(1:nrow(X0Factor),size = 10000,replace = T)
x1Rows <- sample(1:nrow(X1Factor),size = 10000,replace = T)
#X0NumericSample <- X0Numeric[x0Rows,]
#X1NumericSample <- X1Numeric[x1Rows,]
X0FactorSample <- X0Factor[x0Rows,]
X1FactorSample <- X1Factor[x1Rows,]
#numericSample <- rbind(X0NumericSample,X1NumericSample)
factorSample <- rbind(X0FactorSample,X1FactorSample)
factorSample$target <- as.factor(factorSample$target)
#factorSampleWithoutTarget <- factorSample[-ncol(factorSample)]
#numericCorrInSample <- cor(numericSample[-ncol(numericSample)])
#highlyCorrelatedInSample <- findCorrelation(numericCorrInSample, cutoff=0.7)
#linearlyIndependentNumericFeaturesInSample <- numericSample[-highlyCorrelatedInSample]
#numericAndFactorTraining <- cbind(factorSampleWithoutTarget,linearlyIndependentNumericFeaturesInSample)

control <- trainControl(classProbs = T,summaryFunction = twoClassSummary)
#control <- rfeControl(functions=rfFuncs, method="repeatedcv",number=5, repeats=3,classProbs = T,summaryFunction=twoClassSummary)
model <- train(target~., data=factorSample, method="rf", trControl=control,metric = "ROC")
#model <- rfe(target~., data=numericAndFactorTraining, method="rf", trControl=control,metric="Accuracy")

colNamesWOTarget <- colnames(linearlyIndependentNumericFeaturesInSample)
colNamesWOTarget <- colNamesWOTarget[1:(length(colNamesWOTarget)-1)]
submission <- data.frame(ID=test$ID)
test1 <- test[colNamesWOTarget]
test1[(is.na(test1))]<- (-90909090)
submission$target <- NA 
target <- character()
for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
  #submission[rows, "target"] <- predict(model, test1[rows,])
  target <- c(target,as.character(predict(model, test1[rows,])))
}
submission$target <- target
submission[submission$target=="X0",]$target <-0
submission[submission$target=="X1",]$target <-1
# submission[submission$target==1,]$target <-0
# submission[submission$target==2,]$target <-1
outputFile <- ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/results/RFNumericAndFactorEqualRatiosSampleCutOff0P7.csv",
                     "/media/3TB/kag/springleafmarketingresponse/result/RFNumericAndFactorEqualSampleCutOff0P7.csv")
write.csv(submission,file=outputFile,row.names=F)
