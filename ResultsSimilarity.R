probsCutoff <- function(df1,df2){
  m <- merge(df1,df2,by="ID")
  m$target1 <- ifelse(m$target.x<0.5,0,1)
  m$target2 <-ifelse(m$target.y<0.5,0,1)
  m$NotEqual <- m$target1!=m$target2
  return(m)
}
#make f1 the highest score - always
f1 <-"/media/3TB/kag/springleafmarketingresponse/result/XgBTrainModParamsAllFactors_2015-09-2201:34:01_bigtumor_3_submission.csv"
f2 <- "/media/3TB/kag/springleafmarketingresponse/result/XgBTrainDefParamsCV5_2015-09-2115:46:50_bigtumor_3_submission.csv"
merged <- probsCutoff(read.csv(f1),read.csv(f2))
nrow(merged[merged$NotEqual==T,])/nrow(merged)