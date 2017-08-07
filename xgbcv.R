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
load("/media/3TB/kag/springleafmarketingresponse/result/XGBoostAllNumericAndFactorsEQThirdAndUnifCombined_2015-10-0904:52:59_suresh-le_6_WS.rdata")
nround.cv = 10
# ( bst.cv <- xgb.cv(param=param, data=dtrain, 
#                               nfold=10, nrounds=nround.cv, prediction=TRUE, verbose=T) )
paramCV <- list(  objective           = "binary:logistic", 
                #booster = "gblinear",
                #gamma               =10,
                eta                 = 0.01,
                max_depth           = 10,  
                subsample           = 0.7,
                colsample_bytree    = 0.9,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
)
history <-xgb.cv(params = paramCV,data = dtrain, nround=2,  nfold = 5)

save(history,file="/media/3TB/kag/springleafmarketingresponse/result/XgbCVResult.rdata")