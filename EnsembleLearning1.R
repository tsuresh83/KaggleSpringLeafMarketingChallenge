rm(list=ls())
library(doMC)
library(xgboost)
library(caret)
library(pROC)
library(caretEnsemble)
library(randomForest)
library(adabag)
library(plyr)
library(C50)
library(frbs)
library(ada)
set.seed(13)
startTime <- Sys.time()
#0.78899 accuracy
scriptName <-"EnsembleLearning"
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
# for xgboost with caret
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
# end xgboost with caret
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/train.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/train.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/train.rdata")))
training <- training[,2:ncol(training)]
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
helperFunction <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/",
                                         ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse//media/3TB/kag/caretEnsemble-master/R/helper_functions.R",
                                                "/media/3TB/kag/caretEnsemble-master/R/helper_functions.R"))




# names(training)  # 1934 variables
dateCols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169",
              "VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
colTypesAndRanges <- data.frame()
for(i in 1 : ncol(training)){
  col <- colnames(training)[i]
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=","),NUniqueValues=length(unique(training[,i]))))
}
training <- training[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
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
factorFeatures <- colTypesAndRanges[colTypesAndRanges$Type=="factor",]
# for (f in factorFeatures$Name) {
#   if (!(f %in% dateCols)) {
#     #levels <- unique(c(training[[f]], test[[f]]))
#     training[[f]] <- as.integer(training[[f]] )
#     test[[f]]  <- as.integer(test[[f]])
#   }
# }
#get rid of the date columns - to be replaced with the engineered columns
train <- training[-which(colnames(training)%in% dateCols)]
train <- cbind(splitDateFeatures,train)
train[train$target ==0,]$target <- 'X0'
train[train$target ==1,]$target <- 'X1'
train$target <- as.factor(train$target)
trainFeatures <- train[,1:(ncol(train)-1)]
trainClass <- train$target
trainSampleRows <- runif(nrow(trainFeatures)) <= .7
folds=5
repeats=1
my_control <- trainControl(method='cv', number=folds, repeats=repeats, 
                           returnResamp='none', classProbs=TRUE,
                           returnData=FALSE, savePredictions=T, 
                           verboseIter=TRUE, allowParallel=TRUE,
                           summaryFunction=twoClassSummary,
                           index=createMultiFolds(trainClass[trainSampleRows], k=folds, times=repeats))

modelStartTime <- Sys.time()
# xgbModel <- train(target~.,data=train[trainSampleRows,], 
#                   method = modelInfo,
#                   tuneGrid = data.frame(nrounds=1000,max_depth=6,eta=0.1),
#                   metric = "ROC",
#                   trControl = my_control)
adaBag <- train(x=trainFeatures,y=trainClass, 
                  method = "AdaBag",
                  metric = "ROC",
                  trControl = my_control)
adaBagEnd <- Sys.time()
prefix<-paste(scriptName,gsub(" ","",startTime),nodename,ncores,sep="_")
save(list=ls(all.names=T),file= paste(outputFolder,prefix,"_WS.rdata",sep=""),envir = .GlobalEnv)
print(paste("adaBag completed in ",as.numeric((adaBagEnd-modelStartTime),units="mins")))
# rfModel <- train(x=trainFeatures,y=trainClass, 
#                      method = "rf",
#                      metric = "ROC",
#                      trControl = my_control)
# rfEnd <- Sys.time()
# print(paste("rf completed in ",as.numeric((rfEnd-adaBagEnd),units="mins")))
adaBoostModel <- train(x=trainFeatures,y=trainClass, 
                       method = "AdaBoost.M1",
                       metric = "ROC",
                       trControl = my_control)
adaBoostEnd <- Sys.time()

print(paste("adaboost completed in ",as.numeric((adaBoostEnd-adaBagEnd),units="mins")))
#all.models <- list(xgbModel,gfsGccModel,adaBoostModel)
all.models <- list(adaBag,adaBoostModel)
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) max(x$results$ROC)))
all.modelsbkup <- all.models
class(all.models) <-'caretList'
save(list=ls(all.names=T),file= paste(outputFolder,prefix,"_WS.rdata",sep=""),envir = .GlobalEnv)
greedyStart <- Sys.time()
greedy <- caretEnsemble(all.models)
modelEndTime <- Sys.time()
print(paste("greedy ensemble completed in ",as.numeric((modelEndTime-adaBoostEnd),units="mins")))

# model_list <- caretList(
#   x=trainFeatures,y=trainClass, 
#   trControl=my_control,
#   metric="ROC",
#   methodList=c('AdaBoost.M1','C5.0Cost','GFS.GCCL')
# )
print(summary(greedy))


totalModelTime <- as.numeric((modelEndTime-modelStartTime),units="mins")
#prefix <- paste(scriptName,gsub(" ","",modelEndTime),nodename,ncores,sep="_")
#workSpaceFileName <- paste(outputFolder,prefix,"_WS.rdata",sep="")
save(list=ls(all.names=T),file= paste(outputFolder,prefix,"_WS.rdata",sep=""),envir = .GlobalEnv)

#prediction
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
test1 <- test[-c(1,which(colnames(test)%in% dateCols))]
test1 <- cbind(splitDateFeaturesTest,test1)

featureNames <-colnames(train)[1:(ncol(train)-1)]
test1 <- test1[featureNames]
#test1 <- test1[,2:ncol(test1)]
##end engineering


#submission$target <- NA 
for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
  submission[rows, "target"] <- predict(greedy, test1[rows,])
  
}

endTime <- Sys.time()
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
print(paste("total time ",totalScriptTime))
prefix <- paste(scriptName,gsub(" ","",endTime),nodename,ncores,sep="_")
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)