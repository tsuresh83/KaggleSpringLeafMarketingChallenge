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
scriptName<-"CombineH20UncorrelatedNumeric"
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
#find uncorrelated numeric columns
numericCols <- colTypesAndRanges[colTypesAndRanges$Type %in% c("numeric","integer") ,]
numericColsTraining <- training[numericCols$Name]
numericColsWOID <- numericColsTraining[-c(1)]
#numericColsWOID$target <- training$target
#fillNAs <- preProcess(numeric5PercentSample[-c(ncol(numeric5PercentSample))],method="medianImpute")
#trainingWithoutNAs <- predict(fillNAs,numeric5PercentSample)
# fillNAs <- preProcess(numericColsWOID[-c(ncol(numericColsWOID))],method="medianImpute")
# trainingWithoutNAs <- predict(fillNAs,numericColsWOID)
#fillNAs5P <- preProcess(numeric5PercentSample[-c(ncol(numeric5PercentSample))],method="medianImpute")
#trainingWithoutNAs5P <- predict(fillNAs5P,numeric5PercentSample)
numericSample <- numericColsWOID[sample(1:nrow(numericColsWOID),size = 0.2*nrow(numericColsWOID),replace = T),]
numericCorrInSample <- cor(numericSample[-ncol(numericSample)])
highlyCorrelatedInSample <- findCorrelation(numericCorrInSample, cutoff=0.7)
uncorrelatedCols <- colnames(numericSample[-highlyCorrelatedInSample])
#linearlyIndependentNumericFeaturesInSample <- numericSample[-highlyCorrelatedInSample]
#end uncorrelated numeric columns
print("uncorrelated numeric variables found")
train <- training[-which(colnames(training)%in% factorFeatures$Name)]
train <- train[uncorrelatedCols]
train <- cbind(splitDateFeatures,train)
print(paste("number of numeric features",nrow(colTypesAndRanges[colTypesAndRanges$Type %in% c("numeric","integer") ,])))
print(paste("uncorrelated numeric features",length(uncorrelatedCols)))
h <- sample(nrow(train), nrow(train))
partitionLength <- as.integer(length(h)/5)
endIndices <- seq(1,length(h),by=partitionLength)
part1 <- train[h[endIndices[1]:endIndices[2]],]
part2 <- train[h[endIndices[2]:endIndices[3]],]
part3 <- train[h[endIndices[3]:endIndices[4]],]
part4 <- train[h[endIndices[4]:endIndices[5]],]
val <- train[h[endIndices[5]:endIndices[6]],]
gc()
memory <- ifelse(nodename=="bigtumor","128g",
                 ifelse(os=="Darwin","5g",
                        "50g"))
independent <- colnames(train)[1:(ncol(train)-1)]
dependent <- "target"
localH2O = h2o.init(nthreads=-1,max_mem_size = memory,startH2O = F)
print(memory)
part1h20 <- as.h2o(part1,conn=localH2O)
part2h20 <- as.h2o(part2,conn=localH2O)
part3h20 <- as.h2o(part3,conn=localH2O)
part4h20 <- as.h2o(part4,conn=localH2O)
valh20 <- as.h2o(val,conn=localH2O)
part1h20[,c(dependent)] <- as.factor(part1h20[,c(dependent)])
part2h20[,c(dependent)] <- as.factor(part2h20[,c(dependent)])
part3h20[,c(dependent)] <- as.factor(part3h20[,c(dependent)])
part4h20[,c(dependent)] <- as.factor(part4h20[,c(dependent)])
valh20[,c(dependent)] <- as.factor(valh20[,c(dependent)])

my.glm = h2o.glm(x=independent, y=dependent, training_frame=part1h20, validation_frame = part2h20,family="binomial", standardize=T,
                 lambda_search=TRUE)
h2o.saveModel(my.glm,path = outputFolder)

my.dl = h2o.deeplearning(x=independent, y=dependent,epochs=10,seed=13,reproducible=T,training_frame = part2h20,hidden=c(200,200,100),balance_classes =F)
h2o.saveModel(my.dl,path = outputFolder)
my.rf = h2o.randomForest(x=independent, y=dependent, seed=13,build_tree_one_node = T,training_frame=part3h20,balance_classes =T,ntrees = 1000, max_depth=6,binomial_double_trees=T)
h2o.saveModel(my.rf,path = outputFolder)
my.gbm = h2o.gbm(x=independent, y=dependent, training_frame=part4h20,seed=13,balance_classes =F,ntrees = 1200, max_depth=6,learn_rate=0.02)
h2o.saveModel(my.gbm,path = outputFolder)
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
test1 <- test[-c(1,which(colnames(test)%in% factorFeatures$Name))]
test1 <- test1[uncorrelatedCols[1:(length(uncorrelatedCols)-1)]]
test1 <- cbind(splitDateFeaturesTest,test1)

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
predGLM <- as.data.frame(h2o.predict(my.glm,test1))
predDL <- as.data.frame(h2o.predict(my.dl,test1))
predRF <- as.data.frame(h2o.predict(my.rf,test1))
predGBM <- as.data.frame(h2o.predict(my.gbm,test1))
print("target estimated")
endTime <- Sys.time()
prefix <- paste(scriptName,gsub(" ","",startTime),nodename,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"_Predictions.rdata",sep="")
save(predGLM,predDL,predRF,predGBM,file = workSpaceFileName)
allPreds <- cbind(predGLM,predDL,predRF,predGBM)
colnames(allPreds) <- c("PredictGLM","P0GLM","P1GLM","PredictDL","P0DL","P1DL",
                        "PredictRF","P0RF","P1RF","PredictGBM","P0GBM","P1GBM")
p0s <- allPreds[,grepl("P0",colnames(allPreds))]
p0s$Mean <- rowMeans(p0s)
submission$target <- p0s$P0DL
#submission$target <- p0s$Mean
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)
#all combined -0.77
#glm,gbm - 0.75
#glm,dl/ dl/gbm - 0.25
#for(i in 1:4){
predGLMVal<- as.data.frame(h2o.predict(my.glm, valh20))
predDLVal<- as.data.frame(h2o.predict(my.dl, valh20))
predRFVal<- as.data.frame(h2o.predict(my.rf, valh20))
predGBMVal<- as.data.frame(h2o.predict(my.gbm, valh20))
predictions <- predGLMVal$p1  #third column, p1 is P(Y==1)
labels <- as.data.frame(valh20[,c("target")])[,1]
print(cvAUC::AUC(predictions = predictions , labels = labels)) #0.7474598,0.5(non standard)
predictions <- predDLVal$p1
print(cvAUC::AUC(predictions = predictions , labels = labels)) #0.6614806,0.6611688(eq,epoch=100),0.6621787(epoch=100),0.6581071(500,500,epoch=100),0.7102988(epoch=10,200,200,100),0.6888213(epoch=10,200,200,200)
predictions <- predRFVal$p1
print(cvAUC::AUC(predictions = predictions , labels = labels)) #0.7227515,0.7167496(eq)
predictions <- predGBMVal$p1
print(cvAUC::AUC(predictions = predictions , labels = labels)) #0.7657762,0.7522425(eq)
allPredsVal <- cbind(predGLMVal,predDLVal,predRFVal,predGBMVal)
colnames(allPredsVal) <- c("PredictGLM","P0GLM","P1GLM","PredictDL","P0DL","P1DL",
                           "PredictRF","P0RF","P1RF","PredictGBM","P0GBM","P1GBM")
p1sVal <- allPredsVal[,grepl("P1",colnames(allPredsVal))]
p1sVal$Mean <- rowMeans(p1sVal[,c(3,4)])
predictions <- p1sVal$Mean
# Ensemble test AUC 
print(cvAUC::AUC(predictions = predictions , labels = labels))#1,4=0.7665595,#1,3,4=0.7638541,#2,3,4(eq)=0.734202,#3,4(eq)0.7501432
# 0.7888723

#}
# Base learner test AUC (for comparison)
#L <- length(learner)
#auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred1$basepred)[,l], labels = labels)) 
#data.frame(learner, auc)