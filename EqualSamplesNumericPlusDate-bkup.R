rm(list=ls())
library(e1071)
library(caret)
library(RANN)
library(pROC)
library(doMC)

# training <- read.csv("../input/train.csv")
# test  <- read_csv("../input/test.csv")
#test <-read.csv("/media/3TB/kag/springleafmarketingresponse/data/test.csv")
startTime <- Sys.time()
scriptName <-"EqualSamplesNumericPlusDate"
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/train.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata")))
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/test.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/test.rdata")))

ncores <- ifelse(nodename=="bigtumor",15,
                 ifelse(os=="Darwin",2,6))
registerDoMC(ncores)
outputFolder <- ifelse(nodeName=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/results/",ifelse(os=="Darwin"),"/Users/sthiagar/Kaggle/SpringleafMarketingResponse/results/",
                       "/media/3TB/kag/springleafmarketingresponse/results/")
dir.create(outputFolder,recursive=T)
training[(is.na(training))]<- (-90909090)
test[(is.na(test))]<- (-90909090)
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

splitDateFeatures$target <- training$target
splitDateFeatures[splitDateFeatures$target==0,]$target <-"X0"
splitDateFeatures[splitDateFeatures$target==1,]$target <-"X1"
X0Date <- splitDateFeatures[splitDateFeatures$target =="X0",]
X1Date <- splitDateFeatures[splitDateFeatures$target=="X1",]
x0Rows <- sample(1:nrow(X0),size = 10000,replace = T)
x1Rows <- sample(1:nrow(X1),size = 10000,replace = T)
X0Sample <- X0[x0Rows,]
X1Sample <- X1[x1Rows,]
X0DateSample <- X0Date[x0Rows,]
X1DateSample <- X1Date[x1Rows,]
numericSample <- rbind(X0Sample,X1Sample)
dateSample <- rbind(X0DateSample,X1DateSample)
numericCorrInSample <- cor(numericSample[-ncol(numericSample)])
highlyCorrelatedInSample <- findCorrelation(numericCorrInSample, cutoff=0.7)
linearlyIndependentNumericFeaturesInSample <- numericSample[-highlyCorrelatedInSample]
dateAndLINFeatures <- cbind(dateSample[-ncol(dateSample)],linearlyIndependentNumericFeaturesInSample)
control <- trainControl(classProbs = T,summaryFunction = twoClassSummary)
modelStart <- Sys.time()
#model <- train(target~., data=linearlyIndependentNumericFeatures, method="rf", trControl=control,metric = "ROC",preProc=c("center","scale"))
model <- train(target~., data=dateAndLINFeatures, method="rf", trControl=control,metric="ROC")
modelEnd <- Sys.time()
colNamesWOTarget <- colnames(linearlyIndependentNumericFeaturesInSample)
colNamesWOTarget <- colNamesWOTarget[1:(length(colNamesWOTarget)-1)]
#######################
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


##########################
submission <- data.frame(ID=test$ID)
test1 <- test[colNamesWOTarget]
test1[(is.na(test1))]<- (-90909090)
test1 <- cbind(splitDateFeaturesTest,test1)
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
endTime <- Sys.time()
prefix <- paste(scriptName,gsub(" ","",endTime),sep="_")
workSpaceFileName <- paste(outputFolder,prefix,,"_WS.rdata",sep="")

modelingTime <- as.numeric((modelEnd-modelStart),units="mins")
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
outputFileName <- paste(outputFolder,prefix,"_submission.csv")
write.csv(submission,outputFileName,row.names=F)

