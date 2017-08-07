rm(list=ls())
library(Rtsne)
set.seed(13)
startTime <- Sys.time()
scriptName <-"RTSNE"
set.seed(13)
os <- Sys.info()[["sysname"]]
nodename <- Sys.info()[["nodename"]]
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
ncores <- ifelse(nodename=="bigtumor",5,
                 ifelse(os=="Darwin",2,3))
print(paste(nodename,"cores",ncores))
registerDoMC(ncores)
outputFolder <- ifelse(nodename=="bigtumor","/home/tumor/MLExperimental/springleafmarketingresponse/result/",
                       ifelse(os=="Darwin","/Users/sthiagar/Kaggle/SpringleafMarketingResponse/result/",
                              "/media/3TB/kag/springleafmarketingresponse/result/"))
dateCols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169",
              "VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
colTypesAndRanges <- data.frame()
for(i in 1 : ncol(training)){
  col <- colnames(training)[i]
  colTypesAndRanges <- rbind(colTypesAndRanges,data.frame(Name=col,Type=class(training[,i]),Values=paste(unique(training[,i]),collapse=","),NUniqueValues=length(unique(training[,i]))))
}
training <- training[colTypesAndRanges[colTypesAndRanges$NUniqueValues>1,]$Name]
variablesOfInterest <- get(load("/media/3TB/kag/springleafmarketingresponse/result/rfeOptimizedVariables.rdata"))