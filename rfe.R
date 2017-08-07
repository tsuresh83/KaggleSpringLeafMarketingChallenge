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
scriptName <-"RFE"
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
dateOtherFactorsNumericTraining$target <- as.factor(dateOtherFactorsNumericTraining$target)
target0 <- dateOtherFactorsNumericTraining[dateOtherFactorsNumericTraining$target==0,]
target1 <- dateOtherFactorsNumericTraining[dateOtherFactorsNumericTraining$target==1,]
sampleTraining <- rbind(target0[sample(1:nrow(target0),10000,replace = T),],
                        target1[sample(1:nrow(target1),10000,replace = T),])
sampleTraining$target <- as.character(sampleTraining$target)
sampleTraining[sampleTraining$target=='0',]$target <- 'X0'
sampleTraining[sampleTraining$target=='1',]$target <- 'X1'
sampleTraining$target <- as.factor(sampleTraining$target)
#control <- trainControl(classProbs = T,summaryFunction = twoClassSummary)
control <- rfeControl(functions=rfFuncs, method="repeatedcv", repeats=10)
# run the RFE algorithm
modelStart <- Sys.time()
print("model starting...")
print(Sys.time())
results <- rfe(sampleTraining[,1:(ncol(sampleTraining)-1)], sampleTraining[,ncol(sampleTraining)], sizes=seq(50,(ncol(sampleTraining)/2),by=25), rfeControl=control)
modelEnd <- Sys.time()
endTime <- Sys.time()
prefix <- paste(scriptName,gsub(" ","",endTime),nodename,ncores,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"_WS.rdata",sep="")

modelingTime <- as.numeric((modelEnd-modelStart),units="mins")
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)

