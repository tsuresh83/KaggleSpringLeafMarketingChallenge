rm(list=ls())
library(doMC)
library(xgboost)

set.seed(13)

startTime <- Sys.time()
scriptName <-"XgBoost"
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/train.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata")))
print("Training data set loaded...")
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/test.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/test.rdata")))
print("Test data set loaded")
ncores <- ifelse(nodename=="bigtumor",15,
                 ifelse(os=="Darwin",2,6))
registerDoMC(ncores)
outputFolder <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/",
                       ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/result/",
                              "/media/3TB/kag/springleafmarketingresponse/result/"))



feature.names <- colnames(training)[2:(ncol(training))]
training <- training[feature.names]
# names(training)  # 1934 variables
dateCols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169",
              "VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
for (f in feature.names) {
  if (class(training[[f]])=="character" &
      !(f %in% dateCols)) {
    print(f)
    levels <- unique(c(training[[f]], test[[f]]))
    training[[f]] <- as.integer(factor(training[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
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
#get rid of the date columns - to be replaced with the engineered columns
train <- training[-which(colnames(training)%in% dateCols)]
train <- cbind(splitDateFeatures,train)
#cat("sampling training to get around 8GB memory limitations\n")
train <- train[sample(nrow(train), 140000),]
h <- sample(nrow(train), 70000)
val<-train[-h,]
gc()
train <-train[h,]
gc()

dtrain <- xgb.DMatrix(data.matrix(train[,1:(ncol(train)-1)]), label=train$target)




dval <- xgb.DMatrix(data.matrix(val[,1:(ncol(val)-1)]), label=val$target)

watchlist <- list(eval = dval, train = dtrain)

param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.01,
                max_depth           = 6,  
                subsample           = 0.5,
                colsample_bytree    = 1,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 950, # changed from 300
                    verbose             = 2, 
                    #early.stop.round    = 5,
                    watchlist           = watchlist,
                    maximize            = TRUE)
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
test1 <- test[-c(1,which(colnames(test)%in% dateCols))]
test1 <- cbind(splitDateFeaturesTest,test1)

##end engineering

submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(test1[rows,]))
  
}

endTime <- Sys.time()
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
print(paste("total time ",totalScriptTime))
prefix <- paste(scriptName,gsub(" ","",endTime),nodename,ncores,sep="_")
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)