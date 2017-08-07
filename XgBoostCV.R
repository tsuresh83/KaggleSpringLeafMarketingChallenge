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
scriptName<-ifelse(equalSamples,"XgBoostEqualSamples","XgBoostAllFactors3000RoundsDepth20")
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
##to enable parameter sweep of xgboost with caret
modelInfo <- list(label = "eXtreme Gradient Boosting",
                  library = c("xgboost", "plyr"),
                  type = c("Regression", "Classification"),
                  parameters = data.frame(parameter = c('nrounds', 'max_depth', 'eta'),
                                          class = rep("numeric", 3),
                                          label = c('# Boosting Iterations', 'Max Tree Depth', 
                                                    'Shrinkage')),
                  grid = function(x, y, len = NULL) expand.grid(max_depth = seq(1, len),
                                                                nrounds = floor((1:len) * 50),
                                                                eta = .3),
                  loop = function(grid) {     
                    loop <- ddply(grid, c("eta", "max_depth"),
                                  function(x) c(nrounds = max(x$nrounds)))
                    submodels <- vector(mode = "list", length = nrow(loop))
                    for(i in seq(along = loop$nrounds)) {
                      index <- which(grid$max_depth == loop$max_depth[i] & 
                                       grid$eta == loop$eta[i])
                      trees <- grid[index, "nrounds"] 
                      submodels[[i]] <- data.frame(nrounds = trees[trees != loop$nrounds[i]])
                    }    
                    list(loop = loop, submodels = submodels)
                  },
                  fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
                    if(is.factor(y)) {
                      if(length(lev) == 2) {
                        y <- ifelse(y == lev[1], 1, 0) 
                        dat <- xgb.DMatrix(as.matrix(x), label = y)
                        out <- xgb.train(list(eta = param$eta, 
                                              max_depth = param$max_depth), 
                                         data = dat,
                                         nrounds = param$nrounds,
                                         objective = "binary:logistic",
                                         ...)
                      } else {
                        y <- as.numeric(y) - 1
                        dat <- xgb.DMatrix(as.matrix(x), label = y)
                        out <- xgb.train(list(eta = param$eta, 
                                              max_depth = param$max_depth), 
                                         data = dat,
                                         num_class = length(lev),
                                         nrounds = param$nrounds,
                                         objective = "multi:softprob",
                                         ...)
                      }     
                    } else {
                      dat <- xgb.DMatrix(as.matrix(x), label = y)
                      out <- xgb.train(list(eta = param$eta, 
                                            max_depth = param$max_depth), 
                                       data = dat,
                                       nrounds = param$nrounds,
                                       objective = "reg:linear",
                                       ...)
                    }
                    out
                    
                    
                  },
                  predict = function(modelFit, newdata, submodels = NULL) {
                    newdata <- xgb.DMatrix(as.matrix(newdata))
                    out <- predict(modelFit, newdata)
                    if(modelFit$problemType == "Classification") {
                      if(length(modelFit$obsLevels) == 2) {
                        out <- ifelse(out >= .5, 
                                      modelFit$obsLevels[1], 
                                      modelFit$obsLevels[2])
                      } else {
                        out <- matrix(out, ncol = length(modelFit$obsLevels), byrow = TRUE)
                        out <- modelFit$obsLevels[apply(out, 1, which.max)]
                      }
                    }
                    
                    if(!is.null(submodels)) {
                      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
                      tmp[[1]] <- out
                      for(j in seq(along = submodels$nrounds)) {
                        tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
                        if(modelFit$problemType == "Classification") {
                          if(length(modelFit$obsLevels) == 2) {
                            tmp_pred <- ifelse(tmp_pred >= .5, 
                                               modelFit$obsLevels[1], 
                                               modelFit$obsLevels[2])
                          } else {
                            tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), byrow = TRUE)
                            tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 1, which.max)]
                          }
                        }
                        tmp[[j+1]]  <- tmp_pred
                      }
                      out <- tmp
                    }
                    out  
                  },
                  prob = function(modelFit, newdata, submodels = NULL) {
                    newdata <- xgb.DMatrix(as.matrix(newdata))
                    out <- predict(modelFit, newdata)
                    if(length(modelFit$obsLevels) == 2) {
                      out <- cbind(out, 1 - out)
                      colnames(out) <- modelFit$obsLevels
                    } else {
                      out <- matrix(out, ncol = length(modelFit$obsLevels), byrow = TRUE)
                      colnames(out) <- modelFit$obsLevels
                    }
                    out <- as.data.frame(out)
                    
                    if(!is.null(submodels)) {
                      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
                      tmp[[1]] <- out
                      for(j in seq(along = submodels$nrounds)) {
                        tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
                        if(length(modelFit$obsLevels) == 2) {
                          tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
                          colnames(tmp_pred) <- modelFit$obsLevels
                        } else {
                          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), byrow = TRUE)
                          colnames(tmp_pred) <- modelFit$obsLevels
                        }
                        tmp_pred <- as.data.frame(tmp_pred)
                        tmp[[j+1]]  <- tmp_pred
                      }
                      out <- tmp
                    }
                    out  
                  },
                  predictors = function(x, ...) {
                    imp <- xgb.importance(x$xNames, model = x)
                    x$xNames[x$xNames %in% imp$Feature]
                  },
                  varImp = function(object, numTrees = NULL, ...) {
                    imp <- xgb.importance(x$xNames, model = x)
                    imp <- as.data.frame(imp)[, 1:2]
                    rownames(imp) <- as.character(imp[,1])
                    imp <- imp[,2,drop = FALSE]
                    colnames(imp) <- "Overall"
                    imp   
                  },
                  levels = function(x) x$obsLevels,
                  tags = c("Tree-Based Model", "Boosting", "Ensemble Model", "Implicit Feature Selection"),
                  sort = function(x) {
                    # This is a toss-up, but the # trees probably adds
                    # complexity faster than number of splits
                    x[order(x$nrounds, x$max_depth, x$eta),] 
                  })

###end parameter sweep enablers

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
#get rid of the date columns - to be replaced with the engineered columns
#train <- training[-which(colnames(training)%in% factorFeatures$Name)]
#train <- training[-c(which(colnames(training)%in%c("VAR_0200","VAR_0404","VAR_0493","VAR_0214")),which(colnames(training)%in% dateCols))]#var_0200 has 12387 unique values - likely useless
#usefulFactors <- factorFeatures[!(factorFeatures$Name %in% c("VAR_0200","VAR_0404","VAR_0493","VAR_0214",dateCols)),]$Name
usefulFactors <- factorFeatures[!(factorFeatures$Name %in% dateCols),]$Name
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
param <- list(  objective           = "binary:logistic", 
                #booster = "gblinear",
                #gamma               =10,
                eta                 = 0.01,
                max_depth           = 20,  
                #subsample           = 0.7,
                #colsample_bytree    = 0.8,
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
                    nrounds             = 3000, # changed from 300
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
submission <- data.frame(ID=test$ID,target=NA)
#test1 <- test[-c(1,which(colnames(test)%in% dateCols))]
usefulFactorDummiesTest <- dummy(test[which(colnames(test) %in% usefulFactors)])
test1 <- test[-c(1,which(colnames(test)%in% factorFeatures$Name))]
test1 <- cbind(splitDateFeaturesTest,usefulFactorDummiesTest,test1)

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