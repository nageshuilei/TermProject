# 1. according to analysis, both output could be viewed as 0-100 categories. 
# or 0-0.5-1-...-100 categories. 

# 2. thus, we could use logistic regression, svm, and randon forest for prediction. 
#

#library(aod)
#library(ggplot2)

# 3. Install packages: SVM (e1071), Rondom Forest(randomForest), Neural Networks(neuralnet)
# install.packages("e1071")
# install.packages("randomForest")
# install.packages("neuralnet")

library(e1071)
library(randomForest)
library(neuralnet)
traindata <- read.csv("HW3_data/subj2_high_avgrep_train.csv")

#names(traindata)
#head(traindata)
#summary(traindata)

trainfeatures <- subset(traindata, select = -c(1, 2) )
trainoutput1 <- traindata[,1]
trainoutput2 <- traindata[,2]
#names(trainfeatures)
#names(trainoutput1)
#names(trainoutput2)

testdata <- read.csv("HW3_data/subj2_high_avgrep_test.csv")
#names(testdata)

testfeatures <- subset(testdata, select = -c(1, 2) )


## classification
#svmmodel <- svm(trainfeatures,trainoutput1,type='C',kernel='"radial')


# length(trainfeatures)
# trainoutput1
# length(trainoutput1)
rfmodel <- randomForest(trainfeatures, trainoutput1, importance=TRUE, proximity=TRUE)
# print(rfmodel)
testoutput1 <- predict(rfmodel, testfeatures)
head(testoutput1)
write.csv(cbind(Output1=testoutput1, testfeatures),file = "HW3_data/subj2_high_avgrep_rf.csv")

#write.csv(cbind(Output1=testoutput1,Output2=testouput1, testfeatures),file = "HW3_data/subj2_high_avgrep_rf.csv")


