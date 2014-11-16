---
output: word_document
---
###**Project for Practical Machine Learning**



####**Synopsis**

By using data collected from devices such as Jawbone Up, Nike FuelBand, and Fitbit, we built a model to predict the manner in which the participants did the exercise. This dataset includes variables generated from accelerometers on the belt, forearm, arm, and dumbell. There were 6 participants in this study, during which they were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Detailed information is available in this website: http://groupware.les.inf.puc-rio.br/har (the Weight Lifting Exercise Dataset).




####**Loading and Processing the Raw Data**


We first downloaded the training dataset from the website. This dataset includes a header.

```{r, results='hide'}
trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trainurl, "./trainfile.csv", method="curl")
traincsv <- read.csv("./trainfile.csv", header=T)
```



After reading in the training file, we check the dimension, the structures of included variables, and the first few rows in this dataset.

```{r, echo = T, results='hide'}
dim(traincsv)
colnames(traincsv)
str(traincsv)
head(traincsv)
```



There are 160 collected variables and 19622 observations. The variable that we would like to predict is classe. Here we took a lot at the distribution of this variable.

```{r}
table(traincsv$classe)
```



We removed the first 7 variables that are unrelated to the outcome of interest (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window). Some variables have huge amount of "NA" value and empty value. We also excluded these variables from our model building process. In addition, we removed the variables having near zero variances.


```{r}
library(caret)

traincsv <- traincsv[,-c(1:7)] ##removed the first 7 variables

## removed the variables that contain more than 20% NAs or empty value.
nas <- apply(traincsv, 2, function(x) {sum(is.na(x))})
traincsv <- traincsv[,which(nas< nrow(traincsv)*0.8)]
null <- apply(traincsv, 2, function(x) {sum(x=="")})
traincsv <- traincsv[,which(null< nrow(traincsv)*0.8)]

## removed variables with near-zero variance
nzv <- nearZeroVar(traincsv, saveMetrics=T)
traincsv <- traincsv[,!as.logical(nzv$nzv)]

dim(traincsv) ## check the dimension of the cleaned dataset again
```




####**Model Training**



We divided this training data into a training set (70%) and a test set (30%).

```{r}
set.seed(58418)
inTrain <- createDataPartition(traincsv$classe, p= 0.7, list= F)
training<- traincsv[inTrain, ]
testing <- traincsv[-inTrain, ]
```


We first used the following three models to generate the initial predication models: random forest, boosted trees and linear discriminant analysis.

```{r, cache=TRUE, results='hide'}
set.seed(792)

##model training
##random forest
rf <- train(classe ~ ., method= "rf", data= training)
##boosted trees
gbm <- train(classe ~ ., method= "gbm", data= training, verbose=F)
##linear discriminant analysis
lda <- train(classe ~ ., method= "lda", data= training) 

##obtained fitted value on training set
rf_acc<- predict(rf, newdata= training)
gbm_acc<- predict(gbm, newdata= training)
lda_acc<- predict(lda, newdata= training)
```

```{r}
##obtain the accuracy of training set
##random forest
confusionMatrix(rf_acc, training$classe)$overall['Accuracy']
##boosted trees
confusionMatrix(gbm_acc, training$classe)$overall['Accuracy']
##linear discriminant analysis
confusionMatrix(lda_acc, training$classe)$overall['Accuracy']
```

The accuracy for random forest, boosted trees and linear discriminant analysis is 100%, 97.5% and 70.9%, respectively; and the in-sample-error is 0%, 2.5%, 29.1%.



Then we used test set data to obtain the out-of-sample accuracy of these models.

```{r, results='hide'}
##obtained fitted value on testing set
trf_acc<- predict(rf, newdata= testing)
tgbm_acc<- predict(gbm, newdata= testing)
tlda_acc<- predict(lda, newdata= testing)
```

```{r}
##obtain the accuracy by using the test set
##random forest
confusionMatrix(trf_acc, testing$classe)$overall['Accuracy']
##boosted trees
confusionMatrix(tgbm_acc, testing$classe)$overall['Accuracy']
##linear discriminant analysis
confusionMatrix(tlda_acc, testing$classe)$overall['Accuracy']
```

The accuracy for random forest, boosted trees and linear discriminant analysis is 99.22%, 95.71% and 68.39%, respectively; and the out-of-sample error is 0.78%, 4.29%, 31.61%. Therefore, the random forest model has the highest accuracy and the lowest out-of-sample error. This model will be used as the final model. 




####**Prediction**



Finally, we downloaded the testing dataset from the website, and used the random forest model to predict the 20 cases.

```{r, results='hide'}
##Downloaded test dataset
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testurl, "./testfile.csv", method="curl")
testcsv <- read.csv("./testfile.csv", header=T)
```
```{r}
## predict the cases on test dataset
result <- predict(rf, newdata= testcsv)
result
```
