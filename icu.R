library(tidyverse)
library(caret)
library(readr)
library(rpart)
library(rpart.plot)

icu <- read_csv2("/cloud/project/icu/ICU.csv")

#check first rows
head(icu)

#data cleaning

icu$DECEDE <- as_factor(recode(icu$DECEDE, `0` ="survived",`1` ="died"))
icu$SEXE <- as_factor(recode(icu$SEXE, `0` = "female", `1` ="male"))
icu$CHIR_MED[which(icu$CHIR_MED>1)] <- 1
icu$URG_NURG <- as_factor(recode(icu$URG_NURG, `0` = "urgent",`1` = "non-urgent"))
icu$CONSC <- as_factor(recode(icu$CONSC,`0`=  "no coma",`1` = "deep stupor", `2` = "coma"))


#EDA
pairs(icu) ## Hm quite crowdy

#Lets look at the hypothesis: 
#A low Glasgow score means that the patient didn't respond to certain criteria such as consciousness,...
ggplot(icu,aes(GLASGOW,DECEDE))+
  geom_point()


ggplot(icu,aes(FC,DECEDE))+
  geom_point()


ggplot(icu,aes(TA_SYS,DECEDE))+
  geom_point()

ggplot(icu,aes(CHIR_MED,DECEDE))+
  geom_point()

table(icu$DECEDE)

#Simple tree training vs test set
set.seed(45)

#random re-sample
icuRe <- sample(nrow(icu))
icu <- icu[icuRe,]

train <- icu[round(1:(nrow(icu)*0.8)),]
test <- icu[-round(1:(nrow(icu)*0.8)),]

#Sampling using caret
icuCpart <- createDataPartition(icu$DECEDE,
                                times = 1,
                                p = 0.8,
                                list = F)

train_c <- icu[icuCpart,]
test_c <- icu[-icuCpart,]

#simple tree with default cp to 0.001
model_tree_base <- rpart(DECEDE~., data = train, cp = 0.001, minsplit = 20)


#which complexity level returns less error? Optimal Pruning
printcp(model_tree_base) # CP 0.001 & nsplit = 5

#which predictor are more important ?
varImp(model_tree_base)

#plotting
rpart.plot(model_tree_base)

#confusion matrix
pred_tree_base <- as_factor(recode(model_tree_base$y, `1` = "survived", `2` = "died"))
table(train$DECEDE,pred_tree_base)

#ideal pruning model
model_ideal_prun  <- rpart(DECEDE~., data = icu, cp = 0.001, minsplit = 5)

#which predictor are more important ?
varImp(model_ideal_prun)

#plotting
rpart.plot(model_ideal_prun,
           cex = 0.5)

#But what about our test set ? What's the accuracy of the tree when confronted with the test set ?
pred_test_base <- predict(model_tree_base, newdata = test, type = "class")
pred_test_ideal <- predict(model_ideal_prun, newdata = test, type = "class")

#confusionmatrix
t_base <- table(test$DECEDE,pred_test_base)
t_ideal <- table(test$DECEDE,pred_test_ideal)

confusionMatrix(t_base)
confusionMatrix(t_ideal)

#Using caret rf - normal train-test
set.seed(4512)
MyControl <- trainControl(method = "cv",
                          verboseIter = T,
                          classProbs = T)

model_rf <- train(DECEDE ~.,
                  data = train,
                  method = "rf",
                  trControl = MyControl,
                  preProcess = c("center","scale"))

pred_rf <- predict(model_rf, newdata = test, type = "raw")

table_rf <- table(test$DECEDE,pred_rf)

confusionMatrix(table_rf)

#Caret Rf using datapartition
set.seed(458)
MyControl3 <- trainControl(method = "repeatedcv",
                           number = 20,
                           repeats = 5,
                           verboseIter = T,
                           classProbs = T)

model_rf3 <- train(DECEDE ~.,
                   data = train_c,
                   method = "rf",
                   trControl = MyControl3,
                   preProcess = c("center","scale"))

pred_rf3 <- predict(model_rf, newdata = test_c, type = "raw")

table_rf3 <- table(test_c$DECEDE,pred_rf3)

confusionMatrix(table_rf3)



#simple tree with default cp to 0.001
model_tree_c <- rpart(DECEDE~., data = train_c, cp = 0.001, minsplit = 20)


#which complexity level returns less error? Optimal Pruning
printcp(model_tree_c) # CP 0.001 & nsplit = 4

#which predictor are more important ?
varImp(model_tree_c)

#plotting
rpart.plot(model_tree_c)

#confusion matrix
pred_tree_c <- as_factor(recode(model_tree_c$y, `1` = "survived", `2` = "died"))
table(train_c$DECEDE,pred_tree_c)

#ideal pruning model
model_ideal_c  <- rpart(DECEDE~., data = train_c, cp = 0.001, minsplit = 4)

#which predictor are more important ?
varImp(model_ideal_c)

#plotting
rpart.plot(model_ideal_c,
           fallen.leaves = F,
           tweak = 1.20)

#But what about our test set ? What's the accuracy of the tree when confronted with the test set ?
pred_test_c <- predict(model_tree_c, newdata = test_c, type = "class")
pred_ideal_c <- predict(model_ideal_c, newdata = test_c, type = "class")

#confusionmatrix
t_c <- table(test_c$DECEDE,pred_test_c)
t_ideal_c <- table(test_c$DECEDE,pred_ideal_c)

confusionMatrix(t_c)
confusionMatrix(t_ideal_c)


