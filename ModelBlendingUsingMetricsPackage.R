library("Metrics")
#load("/media/3TB/kag/springleafmarketingresponse/result/XGBoostAllNumericAndFactorsEQThirdAndUnifCombined_2015-10-0904:52:59_suresh-le_6_WS.rdata")
y<-val
labels <- y[,c("target")]
#cols <- submission[,2:3]
valPredUnif <- predict(model, data.matrix(y[,-ncol(y)]))
valpredEq <- predict(modelEQ,data.matrix(y[,-ncol(y)]))
valpredThird <- predict(modelThird,data.matrix(y[,-ncol(y)]))
valPreds <- cbind(valPredUnif,valpredEq,valpredThird)
cols <- c(1,2,3)
fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m)
}
fn.opt <- function(pars) {
  -auc(labels, fn.opt.pred(pars, valPreds[,cols]))
}
pars <- rep(1/length(cols),length(cols))
opt.result <- optim(pars, fn.opt, control = list(trace = T))
s <- data.frame(ID=submission$ID)
wts <- opt.result$par
s$target <- apply(submission[,c(2,3,4)],1,function(x){wtd.mean(x,c(wts[1],wts[2],wts[3]))})
write.csv(s,file=paste("/media/3TB/kag/springleafmarketingresponse/result/XgBoostUnifEqThirdMeanMostFactorsUsingOptim",paste(wts,collapse ="_" ),".csv",sep=""),row.names = F)

#train.pred <- fn.opt.pred(opt.result$par, y[,cols])

#test.pred <- fn.opt.pred(opt.result$par, test[,cols])