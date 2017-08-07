#springleaf marketing response - kaggle
rm(list=ls())
library(e1071)
library(caret)
library(RANN)
library(pROC)
library(doMC)
os <- Sys.info()[["sysname"]]
if(os=="Darwin"){
  load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata")
}else{
  load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata")
}
training[(is.na(training))]<- (-90909090)
# file <- ""
# file <- ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.csv",
#                "/media/3TB/kag/springleafmarketingresponse/data/train.csv")
# training <- read.csv(file)
#determing types and ranges of columns
colTypesAndRanges <- data.frame()
nonNA <- data.frame()
for(i in 1 : ncol(training)){
  col <- colnames(training)[i]
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=","),NUniqueValues=length(unique(training[,i]))))
}
output <- ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/ColTypes.csv",
                 "/media/3TB/kag/springleafmarketingresponse/data/ColTypes.csv")
colTypesAndRanges$NCHAR <- nchar(as.character(colTypesAndRanges$Values))
write.csv(colTypesAndRanges,output,row.names=F)
## end col types
charCols <- colTypesAndRanges[colTypesAndRanges$Type=="character",]
numericCols <- colTypesAndRanges[colTypesAndRanges$Type %in% c("numeric","integer") ,]
numericCols <- numericCols[numericCols$NUniqueValues > 1,]
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
numericColsWOID$target <- as.factor(numericColsWOID$target)
#fillNAs <- preProcess(numeric5PercentSample[-c(ncol(numeric5PercentSample))],method="medianImpute")
#trainingWithoutNAs <- predict(fillNAs,numeric5PercentSample)
# fillNAs <- preProcess(numericColsWOID[-c(ncol(numericColsWOID))],method="medianImpute")
# trainingWithoutNAs <- predict(fillNAs,numericColsWOID)
#fillNAs5P <- preProcess(numeric5PercentSample[-c(ncol(numeric5PercentSample))],method="medianImpute")
#trainingWithoutNAs5P <- predict(fillNAs5P,numeric5PercentSample)
numericSample <- numericColsWOID[sample(1:nrow(numericColsWOID),size = 0.2*nrow(numericColsWOID),replace = T),]
numericCorrInSample <- cor(numericSample[-ncol(numericSample)])
highlyCorrelatedInSample <- findCorrelation(numericCorrInSample, cutoff=0.5)
linearlyIndependentNumericFeaturesInSample <- numericSample[-highlyCorrelatedInSample]
control <- trainControl(method="cv", number=3, repeats=3,classProbs = T,summaryFunction = twoClassSummary)
registerDoMC(4)
#model <- train(target~., data=linearlyIndependentNumericFeatures, method="rf", trControl=control,metric = "ROC",preProc=c("center","scale"))
model <- train(target~., data=linearlyIndependentNumericFeaturesInSample, method="rf", trControl=control,metric = "ROC")
# importance <- varImp(model, scale=FALSE)
# plot(importance)
#allCorr <- numericCorr[dim(numericCorr)[1],]
#allCorr[which(abs(allCorr)>0.5)]
