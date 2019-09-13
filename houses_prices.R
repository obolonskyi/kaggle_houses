
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not
pacman::p_load("caret","ROCR","lift","randomForest", "xgboost") #Check, and if needed install the necessary packages

library(dplyr)
library(magrittr)
library(tidyverse) 
library(lubridate)
library(xgboost)


setwd("~/kaggle/houses")
#data <- read.csv("train.csv", na.strings = "")
#data_application <- read.csv("test.csv", na.strings = "")
data_all <- read.csv("all.csv", na.strings = "")

str(data_all)

data_all$LotFrontage <- as.numeric(data_all$LotFrontage)
data_all$MasVnrArea <- as.integer(data_all$MasVnrArea)
data_all$GarageYrBlt <- as.numeric(data_all$GarageYrBlt)
data_all$BsmtFullBath <- as.factor(data_all$BsmtFullBath)
data_all$BsmtHalfBath <- as.factor(data_all$BsmtHalfBath)
data_all$FullBath <- as.factor(data_all$FullBath)
data_all$HalfBath <- as.factor(data_all$HalfBath)
data_all$BedroomAbvGr <- as.factor(data_all$BedroomAbvGr)
data_all$KitchenAbvGr <- as.factor(data_all$KitchenAbvGr)
data_all$Fireplaces <- as.factor(data_all$Fireplaces)
data_all$GarageArea <- as.integer(data_all$GarageArea)
data_all$GarageCars <- as.integer(data_all$GarageCars)
data_all$BsmtFinSF1<- as.integer(data_all$BsmtFinSF1)
data_all$BsmtFinSF2<- as.integer(data_all$BsmtFinSF2)
data_all$BsmtUnfSF<- as.integer(data_all$BsmtUnfSF)
data_all$TotalBsmtSF<- as.integer(data_all$TotalBsmtSF)

str(data_all)

data_train <- data_all[c(1:1460),c(1:81)]
data_testing <- data_all[-c(1:1460),c(1:80)]

str(data_train)

##Data partition

set.seed(77850) 
inTrain <- createDataPartition(y = data_train$SalePrice, p=0.5, list = FALSE)
training <- data_train[ inTrain,]
testing <- data_train[ -inTrain,]

start_time <- Sys.time()
model_forest <- randomForest(SalePrice~ ., data=training, importance=TRUE,proximity=TRUE, type="response")
print(model_forest)
end_time <- Sys.time()
end_time - start_time

plot(model_forest)
importance(model_forest)
varImpPlot(model_forest)

##Random forest predictions Finding predicitons: probabilities and classification Predict probabilities -- an array with 2 columns: for not defaulted (class 0) and for defaulted (class 1)

forest_prices<-predict(model_forest,newdata=testing,type="response") 

predict_application <- predict(model_forest, newdata=data_testing, type="response")


start_time <- Sys.time()

training.x <-model.matrix(SalePrice~ ., data = training)
testing.x <- model.matrix(SalePrice~ ., data = testing)

model_XGboost<-xgboost(data = data.matrix(training.x[,-1]), 
                       label = training$SalePrice,
                       max_depth = 20, 
                       nround=10000, 
                       booster = "gblinear",
                       objective = "reg:linear")

end_time <- Sys.time()
end_time - start_time


##Predict classification (for confusion matrix)

XGboost_prediction<-predict(model_XGboost,newdata=testing.x[,-1])
XGboost_prediction

data_xgb_application <- model.matrix(SalePrice~ ., data = data_testing)

xgboost_application <- predict(model_XGboost, newdata=data_xgb_application)


write.csv(predict_application, file="submission.csv")


