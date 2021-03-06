library(e1071)
library(lubridate)

train <- read.csv("train.csv",header=TRUE)
test <- read.csv("test.csv",header=TRUE)

train$day<-as.factor(day(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$month<-as.factor(month(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$year<-as.factor(year(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))

test$day<-as.factor(day(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$month<-as.factor(month(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$year<-as.factor(year(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))

train_cols<-train[,c(3:42,44:46)]
labels<-as.matrix(train[,43])
testdata<-test[,3:45]

train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))

fit<- svm(x=as.matrix(train_cols),y=labels,cost=10,scale=TRUE,type="eps-regression")
predictions<-as.data.frame(predict(fit,newdata=testdata))

submit<-as.data.frame(cbind(test[,1],predictions))
colnames(submit)<-c("Id","Prediction")
)
write.csv(submit,"submission.csv",row.names=FALSE,quote=FALSE)
