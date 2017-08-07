rm(list=ls())
library(h2o)
library(h2oEnsemble)
library(SuperLearner)
library(cvAUC)
startTime <- Sys.time()
scriptName<-"H20Ensemble"
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
training <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/data/TrainAndEvaluationWithFactors.rdata",
                   ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/TrainAndEvaluationWithFactors.rdata",
                          "/media/3TB/kag/springleafmarketingresponse/data/TrainAndEvaluationWithFactors.rdata"))
testing <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/data/test1WithNotAllFactorCols.rdata",
               ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test1WithNotAllFactorCols.rdata",
                      "/media/3TB/kag/springleafmarketingresponse/data/test1WithNotAllFactorCols.rdata"))
outputFolder <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/",
                       ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/result/",
                              "/media/3TB/kag/springleafmarketingresponse/result/"))
memory <- ifelse(nodename=="bigtumor","128g",
                       ifelse(os=="Darwin","5g",
                              "50g"))
load(training)
independent <- colnames(train)[1:(ncol(train)-1)]
dependent <- "target"
# Create a custom base learner library & specify the metalearner
metalearner <- c("SL.glm")
#train$target <- as.factor(train$target)
#val$target <- as.factor(val$target)
localH2O = h2o.init(nthreads=-1,max_mem_size = memory)
print(memory)
#tumorh20 <-
# Run regression GBM on australia.hex data
trainh20 <- as.h2o(train,conn=localH2O)
valh20 <- as.h2o(val,conn=localH2O)
trainh20[,c(dependent)] <- as.factor(trainh20[,c(dependent)])
valh20[,c(dependent)] <- as.factor(valh20[,c(dependent)])

#h2o.randomForest.1 <- function(..., ntrees = 500, max_depth=20,binomial_double_trees=T) h2o.randomForest.wrapper(..., ntrees = ntrees, max_depth=max_depth,binomial_double_trees=binomial_double_trees)
#h2o.randomForest.2 <- function(..., ntrees = 1000, max_depth=10,binomial_double_trees=T) h2o.randomForest.wrapper(..., ntrees = ntrees, max_depth=max_depth,binomial_double_trees=binomial_double_trees)
h2o.gbm.1 <- function(..., ntrees = 500, max_depth=20,min_rows=50,learn_rate=0.01) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth=max_depth,min_rows=min_rows,learn_rate=0.01)
#h2o.gbm.2 <- function(..., ntrees = 3000, max_depth=6,min_rows=50) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth=max_depth,min_rows=min_rows)
h2o.gbm.2 <- function(..., ntrees = 3000, max_depth=6,min_rows=50,learn_rate=0.01) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth=max_depth,min_rows=min_rows,learn_rate = 0.01)
#learner <- c("h2o.randomForest.1", "h2o.randomForest.2", "h2o.gbm.1","h2o.gbm.2")
learner <- c("h2o.randomForest.1","h2o.gbm.1")
metalearner <- c("SL.glm")
family <- "binomial"
modelStart <- Sys.time()
fit <- h2o.ensemble(x = independent, y = dependent, 
                    training_frame = trainh20, 
                    validation_frame = valh20,
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner)#,
                    #cvControl = list(V = 2, shuffle = TRUE))
modelEnd <- Sys.time()
totalModelTime <- as.numeric((modelEnd-modelStart),units="mins")
print(paste("Total model time ",totalModelTime))
prefix <- paste(scriptName,gsub(" ","",startTime),nodename,sep="_")
workSpaceFileName <- paste(outputFolder,prefix,"_WS.rdata",sep="")
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
#h2o.saveModel(fit,path=paste(outputFolder,prefix,"_H20Model.rdata",sep=""),force = T)
load(testing)
print("loaded test")
ifelse(nodename=="bigtumor",load("/home/tumor/MLExperimental/springleafmarketingresponse/data/test.rdata"),
       ifelse(os=="Darwin",
              load("/Users/sthiagar/Kaggle/SpringleafMarketingResponse/data/test.rdata"),
              load("/media/3TB/kag/springleafmarketingresponse/data/test.rdata")))
submission <- data.frame(ID=test$ID,target=NA)
# target <- numeirc()
# for (rows in split(1:nrow(test1), ceiling((1:nrow(test1))/10000))) {
#   target <- c(target,(predict(fit, test1[rows,])))
#   
# }
test1 <- as.h2o(test1,conn=localH2O)
pred <- predict.h2o.ensemble(fit,test1)
print("target estimated")
#predictions <- (pred$pred)

submission$target <- pred$pred
endTime <- Sys.time()
totalScriptTime <- as.numeric((endTime-startTime),units="mins")
print(paste("total time ",totalScriptTime))
submissionFileName <- paste(outputFolder,prefix,"_submission.csv",sep="")
write.csv(submission,submissionFileName,row.names=F)
save(list=ls(all.names=T),file=workSpaceFileName,envir = .GlobalEnv)
#diagnostics....
pred1<- predict.h2o.ensemble(fit, valh20)
predictions <- as.data.frame(pred1$pred)  #third column, p1 is P(Y==1)
labels <- as.data.frame(valh20[,c("target")])[,1]


# Ensemble test AUC 
cvAUC::AUC(predictions = predictions , labels = labels)
# 0.7888723


# Base learner test AUC (for comparison)
L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred1$basepred)[,l], labels = labels)) 
data.frame(learner, auc)
h2o.shutdown(localH2O)