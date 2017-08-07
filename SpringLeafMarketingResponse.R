#springleaf marketing response - kaggle
rm(list=ls())
library(caret)
library(foreach)
training <- read.csv("/media/3TB/kag/springleafmarketingresponse/data/train.csv")
#determing types and ranges of columns
colTypesAndRanges <- data.frame()
for(i in 1: ncol(training)){
  col <- colnames(training)[i]
#   nonNA <- training[col]
#   nonNA <- nonNA[!is.na(nonNA)]
#   nonNA <- nonNA[!is.nan(nonNA)]
#   if(is.numeric(nonNA)){
#     nonNA <- nonNA[is.finite(nonNA)]
#   }
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=",")))
}
colTypesAndRanges$NCHAR <- nchar(as.character(colTypesAndRanges$Values))
write.csv(colTypesAndRanges,"/media/3TB//kag/springleafmarketingresponse/ColTypes.csv",row.names=F)
## end col types
#colTypesAndRanges <- read.csv("/media/3TB//kag/springleafmarketingresponse/ColTypes.csv")
charCols <- colTypesAndRanges[colTypesAndRanges$Type=="character",]$Name
numericCols <- colTypesAndRanges[colTypesAndRanges$Type %in% c("numeric","integer"),]$Name
factorCols <- colTypesAndRanges[colTypesAndRanges$Type %in% c("factor"),]$Name
dateCols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169",
              "VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
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
numericColsWOIDAndTarget <- numericCols[2:(length(numericCols)-1)]
numericColsWOID <- numericCols[2:length(numericCols)]
# numericCorrelation <- cor(training[numericColsWOIDAndTarget],use="complete.obs")
# highlyCorrelated <- findCorrelation(numericCorrelation, cutoff=0.1)#not working with lots of NAs
numericTrainingWOID <- training[numericColsWOID]
numericTrainingWOID$target <- as.factor(numericTrainingWOID$target)
numeric5PercentSample <- numericTrainingWOID[sample(1:nrow(numericTrainingWOID),0.05*nrow(numericTrainingWOID),replace = T),]
# control <- trainControl(method="repeatedcv", number=3, repeats=2)
# #sample10Percent <- numericTrainingWOID[sample(1:nrow(numericTrainingWOID),size = 0.5*nrow(numericTrainingWOID),replace = T),]
# model <- train(target~., data=numericTrainingWOID, preProcess=c("scale","center","medianImpute"),method="AdaBag", trControl=control)
control <- trainControl(method="cv", number=3, repeats=3)
# foreach (i = 1:ncol(numeric5PercentSample)) %do%{
#   print(class(numeric5PercentSample[,i]))
# }
preProc <- preProcess(numeric5PercentSample[-c(ncol(numeric5PercentSample))],method = "medianImpute")
training <- predict(preProc,numeric5PercentSample)
model <- train(target~., data=training, method="lvq", trControl=control)