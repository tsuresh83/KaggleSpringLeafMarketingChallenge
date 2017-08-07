rm(list=ls())
library(e1071)
library(caret)
library(RANN)
library(pROC)
library(doMC)
library(caTools)
# training <- read.csv("../input/train.csv")
# test  <- read_csv("../input/test.csv")
#test <-read.csv("/media/3TB/kag/springleafmarketingresponse/data/test.csv")
os <- Sys.info()[["sysname"]]
ifelse(os=="Darwin",load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata"))
ifelse(os=="Darwin",load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),load("/media/3TB/kag/springleafmarketingresponse/data/test.rdata"))
ifelse(os=="Darwin",registerDoMC(2),registerDoMC(6))
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
for(col in colnames(dateColsAsDate)){
  temp <- as.vector(dateColsAsDate[col])
  dateColsAsDate[col] <- as.Date(temp[,1],"%d%b%y")
}

stateCols <- c("VAR_0274","VAR_0237")
designationCols <- c("VAR_0493","VAR_0404") # seem like the designations of the potential customer
dataSource <- c("VAR_1934")
logicalCols <- c("VAR_0236","VAR_0232","VAR_0230","VAR_0226")
#logical is useless - always 'false' or 'NA'
numericColsTraining <- training[numericCols$Name]
numericColsTrainingWOIDAndTarget <- numericColsTraining[-c(1,ncol(numericColsTraining))]
numericColsWOID <- numericColsTraining[-c(1)]
numericColsWOID$target <- training$target
numericColsWOID[numericColsWOID$target==0,]$target <-"X0"
numericColsWOID[numericColsWOID$target==1,]$target <-"X1"
numericColsWOID$target <- as.factor(numericColsWOID$target)
X0 <- numericColsWOID[numericColsWOID$target=="X0",]
X1 <- numericColsWOID[numericColsWOID$target=="X1",]
X0Sample <- X0[sample(1:nrow(X0),size = 10000,replace = T),]
X1Sample <- X1[sample(1:nrow(X1),size = 10000,replace = T),]
numericSample <- rbind(X0Sample,X1Sample)
numericCorrInSample <- cor(numericSample[-ncol(numericSample)])
highlyCorrelatedInSample <- findCorrelation(numericCorrInSample, cutoff=0.9)
linearlyIndependentNumericFeaturesInSample <- numericSample[-highlyCorrelatedInSample]
control <- trainControl(classProbs = T,summaryFunction = twoClassSummary)
#LogitBoost 0.5 roc tops (upto 150 nIter)
#Adaboost 0.79 ROC with mfinal=100 and maxDepth=30
#model <- train(target~., data=linearlyIndependentNumericFeatures, method="rf", trControl=control,metric = "ROC",preProc=c("center","scale"))
#model <- train(target~., data=linearlyIndependentNumericFeaturesInSample, method="rf", trControl=control,metric="Accuracy")
#model <- train(target~., data=linearlyIndependentNumericFeaturesInSample, method="LogitBoost", trControl=control,metric="ROC",tuneGrid=data.frame(nIter=seq(10,150,10)),preProc=c("center","scale"))
model <- train(target~., data=linearlyIndependentNumericFeaturesInSample, method="AdaBag", trControl=control,metric="ROC",tuneGrid=expand.grid(mfinal=seq(100,400,50),maxdepth=seq(30,80,5)))
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
ifelse(os=="Darwin",write.csv(submission,"/Users/sthiagar/Kaggle/SpringleafMarketingResponse/results/RFNumericFeaturessubmissionEqualRatiosSampleCutOff0P7AdaBoost_mFinal100_maxDepth30.csv",row.names=F),
       write.csv(submission, "/media/3TB/kag/springleafmarketingresponse/result/RFNumericFeaturessubmissionEqualRatiosSampleCutOff0P7AdaBoost_mFinal100_maxDepth30.csv",row.names=F))

