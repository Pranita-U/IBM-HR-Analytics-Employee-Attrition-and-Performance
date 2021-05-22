install.packages('entropy', dependencies=TRUE)
install.packages('corrplot', dependencies=TRUE)


library(data.table)
library(entropy)
library(ggplot2)
library(caTools)
library(ROCR)
library(rpart)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
library(corrplot)
library(pROC)






getwd()
setwd("C:\\Users\\Pranita\\Desktop\\Imarticus\\Project")
getwd()

AttrData=read.csv("AttritionData.csv",na.strings = "")
View(AttrData)
dim(AttrData)

set.seed(1) # This is used to reproduce the same composition of the sample #
RowNumbers = sample(x=1:nrow(AttrData),size = 0.70*nrow(AttrData))
View(RowNumbers)
head(AttrData)
TrainData = AttrData[RowNumbers,]#Trainset
TestData = AttrData[-RowNumbers,]#Testset
View(TrainData)
View(TestData)
dim(TrainData)
dim(TestData)


#Add a source column to train set and test set and call in "Train" and "Test" respectively
TrainData$Source="Train"
TestData$Source="Test"
View(TrainData)
View(TestData)
dim(TrainData)
dim(TestData)


#Combine the Train set and Test set as Full data
#this full data is used for data prep and modelling
Fulldata=rbind(TrainData,TestData)
View(Fulldata)
dim(Fulldata)

#Check for NA values
sum(is.na(Fulldata))
colSums(is.na(Fulldata))
summary(Fulldata)

#Checking the classes of columns
str(Fulldata)

#get the categorical variables
## converting categorical columns to factors
cat_vari=c("Attrition","BusinessTravel","Department","Education","EducationField","EnvironmentSatisfaction","Gender",
           "JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","Over18","OverTime",
           "PerformanceRating","RelationshipSatisfaction","StandardHours","StockOptionLevel","WorkLifeBalance")
factors1.df=Fulldata[,cat_vari]
View(factors1.df)
length(cat_vari)

#or
#is.factor=sapply(Fulldata,is.factor)
#factors.df=Fulldata[,is.factor]
#View(factors.df)
#length(factors.df)
#cat_var=Fulldata[which(Fulldata[class("chr")])]


# Checking summary of data to know about the distribution of data
summary(Fulldata)


### creating Dummy variables###
Dummy_df = model.matrix(~ BusinessTravel + Department + Education + EducationField + EnvironmentSatisfaction + 
                          Gender + JobInvolvement + JobLevel + JobRole + JobSatisfaction + MaritalStatus + OverTime + 
                          PerformanceRating + RelationshipSatisfaction + StockOptionLevel + WorkLifeBalance, data = Fulldata)
View(Dummy_df)
nrow(Dummy_df)
ncol(Dummy_df)
dim(Fulldata)
dim(Dummy_df)
Fulldata2 = cbind(Fulldata,Dummy_df[,-1])
View(Fulldata2)
dim(Fulldata2)

Fulldata3=subset(Fulldata2, select=-c(BusinessTravel,Department,Education,EducationField,EnvironmentSatisfaction, 
                                        Gender,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,OverTime,
                                        PerformanceRating,RelationshipSatisfaction,StockOptionLevel,WorkLifeBalance,EmployeeCount,
                                        Over18,StandardHours))
View(Fulldata3)
dim(Fulldata3)

#Convert dependent variable into 0s and 1s 
Fulldata3$Attrition = ifelse(Fulldata3$Attrition == "Yes",1,0)
View(Fulldata3)
dim(Fulldata3)
names(Fulldata3)
names(Fulldata3) = gsub(" ", "_", names(Fulldata3))
names(Fulldata3)

colnames(Fulldata3)[20] = "Dep_Res_Dev"
names(Fulldata3)


#Divide the data into train and test data based on the Source column
#and make sure that the source column shd be dropped
Train = subset(Fulldata3,subset = (Source == "Train"), select = -Source)
Test = subset(Fulldata3,subset = (Source == "Test"), select = -Source)

View(Train)
dim(Train)


View(Test)
dim(Test)



#Check the proportion of 0s and 1s in the dep vari
round(table(Train$Attrition)/nrow(Train)*100,2)



#Multicolinearity check
library(caTools)
M0 = lm(Attrition~.,data=Train)
vif(M0)


#Build logistic regn model
M1 = glm(Attrition~.,data = Train,family = "binomial")
summary(M1)
length(M1)


#any(is.na("EmployeeCount"))
#length(which(is.na("EmployeeCount")))

#Removing and updation of our model by rmvg the most insignificant ind vari
M2 = update(M1, . ~ . -StockOptionLevel)
summary(M2)


M3= update(M2, . ~ . -DepartmentSales)
summary(M3)


M4= update(M3, . ~ . -MonthlyIncome)
summary(M4)


M5= update(M4, . ~ . -HourlyRate)
summary(M5)


M6= update(M5, . ~ . -`JobRoleResearch_Director`)
summary(M6)

M7= update(M6, . ~ . -`EducationFieldTechnical_Degree`)
summary(M7)


M8= update(M7, . ~ . -PerformanceRating)
summary(M8)


M9= update(M8, . ~ . -`JobRoleManufacturing_Director`)
summary(M9)


M10= update(M9, . ~ . -`JobRoleResearch_Scientist`)
summary(M10)


M11= update(M10, . ~ . -PercentSalaryHike)
summary(M11)


M12= update(M11, . ~ . -EmployeeNumber)
summary(M12)


M13= update(M12, . ~ . -DailyRate)
summary(M13)


M14= update(M13, . ~ . -TotalWorkingYears)
summary(M14)


M15= update(M14, . ~ . -EducationFieldMarketing)
summary(M15)


M16= update(M15, . ~ . -Education)
summary(M16)


M17= update(M16, . ~ . -EducationFieldOther)
summary(M17)


M18= update(M17, . ~ . -JobRoleManager)
summary(M18)


M19= update(M18, . ~ . -Dep_Res_Dev)
summary(M19)


M20= update(M19, . ~ . -MonthlyRate)
summary(M20)


M21= update(M20, . ~ . -MaritalStatusMarried)
summary(M21)


M22= update(M21, . ~ . -TrainingTimesLastYear)
summary(M22)



#Predict on Trainset data using the final model M22
Train_Prob = predict(M22, Train, type = "response")
head(Train_Prob)


Train_Class = ifelse(Train_Prob>0.5,1,0)
head(Train_Class)
View(Train_Class)

table(Train_Class,Train$Attrition)



#Predict on Testset
Test_Prob = predict(M22,Test,type = "response")
head(Test_Prob)

Test_Class = ifelse(Test_Prob>0.5,1,0)
head(Test_Class)
View(Test_Class)

table(Test_Class,Test$Attrition)


library(ROCR)


ROC_pred = prediction(Train_Prob,Train$Attrition)
#Prediction produces an output whc has attributes like TP,FP,TN,etc


ROC_curve = performance(ROC_pred,"tpr", "fpr")
#performance():all kinds of predictor evaluations like tpr, fpr,accu,etc are performed using
#this func
#It helps in creating the ROC curve

#ROC curve
windows()
plot(ROC_curve)



Cutoff_Tbale = cbind.data.frame(Cutoff = ROC_curve@alpha.values[[1]],
                                FPR = ROC_curve@x.values[[1]],
                                TPR = ROC_curve@y.values[[1]])

View(Cutoff_Tbale)



"Attrition"
table(AttrData$Attrition)

print("Confusion matrix for threshold 0.5")

thershold= 0.5

confusion_mat = table(Test$Attrition, Test_Prob > thershold)
confusion_mat
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- confusion_mat[4]
tp_plus_fn <- confusion_mat[4] + confusion_mat[2]

sensitivity <- tp/tp_plus_fn
print(c("sensitivity",sensitivity))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- confusion_mat[1] + confusion_mat[3]

specificity <- tn/tn_plus_fp
print(c("specificity",specificity))

#FPR
FP_Rate = 1- specificity
print(c("False positive rate", FP_Rate))


# accuracy
Accuracy_Value = (confusion_mat[1] + confusion_mat[4])/(confusion_mat[1]+confusion_mat[2]+confusion_mat[3]+confusion_mat[4])
print(c("Accuracy",Accuracy_Value))

#Precision
Precision_value=(confusion_mat[4])/(confusion_mat[4]+confusion_mat[3])
print(c("Precision_value", Precision_value))

#Recall
Recall_value=(confusion_mat[4])/(confusion_mat[4]+confusion_mat[2])
print(c("Recall_value", Recall_value))


#F1 Score
F1_Score= (2*Precision_value*Recall_value)/(Precision_value+Recall_value)
print(c("F1 score", F1_Score))
####################################################

print("Confusion matrix for threshold 0.7")

thershold= 0.7

confusion_mat = table(Test$Attrition, Test_Prob > thershold)
confusion_mat
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- confusion_mat[4]
tp_plus_fn <- confusion_mat[4] + confusion_mat[2]

sensitivity <- tp/tp_plus_fn
print(c("sensitivity",sensitivity))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- confusion_mat[1] + confusion_mat[3]

specificity <- tn/tn_plus_fp
print(c("specificity",specificity))

#FPR
FP_Rate = 1- specificity
print(c("False positive rate", FP_Rate))


# accuracy
Accuracy_Value = (confusion_mat[1] + confusion_mat[4])/(confusion_mat[1]+confusion_mat[2]+confusion_mat[3]+confusion_mat[4])
print(c("Accuracy",Accuracy_Value))

#Precision
Precision_value=(confusion_mat[4])/(confusion_mat[4]+confusion_mat[3])
print(c("Precision_value", Precision_value))

#Recall
Recall_value=(confusion_mat[4])/(confusion_mat[4]+confusion_mat[2])
print(c("Recall_value", Recall_value))


#F1 Score
F1_Score= (2*Precision_value*Recall_value)/(Precision_value+Recall_value)
print(c("F1 score", F1_Score))
#####################################################

print("Confusion matrix for threshold 0.1")

thershold= 0.1

confusion_mat = table(Test$Attrition, Test_Prob > thershold)
confusion_mat
# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- confusion_mat[4]
tp_plus_fn <- confusion_mat[4] + confusion_mat[2]

sensitivity <- tp/tp_plus_fn
print(c("sensitivity",sensitivity))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- confusion_mat[1] + confusion_mat[3]

specificity <- tn/tn_plus_fp
print(c("specificity",specificity))

#FPR
FP_Rate = 1- specificity
print(c("False positive rate", FP_Rate))


# accuracy
Accuracy_Value = (confusion_mat[1] + confusion_mat[4])/(confusion_mat[1]+confusion_mat[2]+confusion_mat[3]+confusion_mat[4])
print(c("Accuracy",Accuracy_Value))

#Precision
Precision_value=(confusion_mat[4])/(confusion_mat[4]+confusion_mat[3])
print(c("Precision_value", Precision_value))

#Recall
Recall_value=(confusion_mat[4])/(confusion_mat[4]+confusion_mat[2])
print(c("Recall_value", Recall_value))


#F1 Score
F1_Score= (2*Precision_value*Recall_value)/(Precision_value+Recall_value)
print(c("F1 score", F1_Score))


#########################
## Plotting Receiver operator characteristics curve to decide better on threshold
rocr_pred_logistic_best_treshold = prediction(Test_Prob ,Test$Attrition)
rocr_perf_logistic_best_treshold = performance(rocr_pred_logistic_best_treshold,'tpr','fpr')
windows()
plot(rocr_perf_logistic_best_treshold,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))


thershold_best_log = 0.3

conf_mat_logistic_best_treshold <- table(Test$Attrition ,Test_Prob > thershold_best_log)
# accuracy
accuracy_logistic_best_treshold <- (conf_mat_logistic_best_treshold[1] + conf_mat_logistic_best_treshold[4])/(conf_mat_logistic_best_treshold[1]+conf_mat_logistic_best_treshold[2]+conf_mat_logistic_best_treshold[3]+conf_mat_logistic_best_treshold[4])
"Confusion matrix for best threshold (logistic regression)"
conf_mat_logistic_best_treshold
"Model Performance"
print(c("Accuracy",accuracy_logistic_best_treshold))

# sensitivity tpr --> sensitivity = tp/(tp+FN)
tp <- conf_mat_logistic_best_treshold[4]
tp_plus_fn <- conf_mat_logistic_best_treshold[4] + conf_mat_logistic_best_treshold[2]

sensitivity_logistic_best_treshold <- tp/tp_plus_fn
print(c("sensitivity",sensitivity_logistic_best_treshold))

# specificity tnr--> specificity = tn/(tn+FP)
tn <- confusion_mat[1]
tn_plus_fp <- conf_mat_logistic_best_treshold[1] + conf_mat_logistic_best_treshold[3]

specificity_logistic_best_treshold <- tn/tn_plus_fp
print(c("specificity",specificity_logistic_best_treshold))

#FPR
FP_Rate = 1- specificity_logistic_best_treshold
print(c("False positive rate", FP_Rate))


#Precision
Precision_value=(conf_mat_logistic_best_treshold[4])/(conf_mat_logistic_best_treshold[4]+conf_mat_logistic_best_treshold[3])
print(c("Precision_value", Precision_value))

#Recall
Recall_value=(conf_mat_logistic_best_treshold[4])/(conf_mat_logistic_best_treshold[4]+conf_mat_logistic_best_treshold[2])
print(c("Recall_value", Recall_value))


#F1 Score
F1_Score= (2*Precision_value*Recall_value)/(Precision_value+Recall_value)
print(c("F1 score", F1_Score))




install.packages('e1071', dependencies=TRUE)
library(e1071)
install.packages('caret', dependencies=TRUE)
library(caret)




##############################################################################################################
#SVM Model building
install.packages('e1071', dependencies=TRUE)
library(e1071)


SVM_M1 = svm(as.factor(Attrition)~ .,kernel = "linear", data = Train)
SVM_M1$index
summary(SVM_M1)

#Prediction on test
SVM_M1_Test_Pred = predict(SVM_M1,Test)

#Confusion matrix
table(SVM_M1_Test_Pred,Test$Attrition)

#Accuracy
sum(diag(table(SVM_M1_Test_Pred,Test$Attrition)))/nrow(Test)

#ROC_Curve
#rocr_pred_svm = prediction(M1_Test_Pred,Testset$Attrition)
#rocr_perf_svm = performance(rocr_pred_svm,'tpr','fpr')
#plot(rocr_perf_svm,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))
#plot(M1,Test)

#SVM Model building framework
my_kernel = c("linear", "radial","polynomial")
my_cost = c(0.1,0.2,0.3)
my_gamma = c(0,0.2,0.3)

for (i in my_kernel)
{
  for (j in my_cost)
  {
    for (k in my_gamma)
    {
      
      Model_svm = svm(as.factor(Attrition)~ .,kernel = i,cost = j,gamma = k, data = Train)
      M_Test_Pred = predict(Model_svm,Test)
      
      #Confusion matrix
      table(M_Test_Pred,Test$Attrition)
      
      #Accuracy
      accuracy_model = sum(diag(table(M_Test_Pred,Test$Attrition)))/nrow(Test)
      
      #o/p showing kernel,cost,gamma,accu values
      My_svm_model = cbind(my_kernel,my_cost,my_gamma,accuracy_model)
    }
  }
}

View(My_svm_model)
head(My_svm_model)

###########################################################################################################

#DT_RF
#CART Model

install.packages("randomForest")
library(randomForest)
install.packages("rpart")
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)

set.seed(123)
DT_M1 = rpart(Attrition ~., method = "class", data=Train)


# DT Plot
library(rpart.plot)
windows()
rpart.plot(DT_M1, cex = 0.65) # cex is character expansion


# Prediction on testing set
DT_Test_Pred = predict(DT_M1, Test, type = "vector")


#ROC_Curve
rocr_pred_cart = prediction(DT_Test_Pred ,Test$Attrition)
rocr_perf_cart = performance(rocr_pred_cart,'tpr','fpr')
windows()
plot(rocr_perf_cart,colorize=TRUE,print.cutoffs.at = seq(0,1,.1),text.adj =c(-0.2,1.7))

# Confusion Matrix
table(Test$Attrition, DT_Test_Pred)

# Accuracy
sum(diag(table(Test$Attrition, DT_Test_Pred)))/nrow(Test)


#######################
##RF
#######################
library(randomForest)
library(caret)
set.seed(123)

#Telling R to perform Classification instead of Regression
Train$Attrition <- as.factor(Train$Attrition)
Test$Attrition <- as.factor(Test$Attrition)
#RF_Model1 = train(Attrition~.,data=Trainset,method="rf", trControl=trainControl(method = "oob", number = 3))
RF_M1 = randomForest(Attrition ~ .,data = Train, ntree = 500, mtry = 2,importance = TRUE)
RF_M1

#Predict on Test
RF_Test_Pred1 = predict(RF_M1,Test,type="class")

#confusion matrix
table(Test$Attrition,RF_Test_Pred1)
mean(RF_Test_Pred1 == Test$Attrition)

#Accuracy
sum(diag(table(Test$Attrition,RF_Test_Pred1)))/nrow(Test)

#Variable importance plot
importance(RF_M1)
windows()
VarImp1 = varImpPlot(RF_M1)
View(VarImp1) #to view the values in a table



RF_M2 = randomForest(Attrition ~ .,data = Train, ntree = 500, mtry = 6,importance = TRUE)
RF_M2

#Predict on Test
RF_Test_Pred2 = predict(RF_M2,Test,type="class")

#confusion matrix
table(Test$Attrition,RF_Test_Pred2)
mean(RF_Test_Pred2 == Test$Attrition)

#Accuracy
sum(diag(table(Test$Attrition,RF_Test_Pred2)))/nrow(Test)


#Variable importance plot
importance(RF_M2)
windows()
VarImp2 = varImpPlot(RF_M2)
View(VarImp2) #to view the values in a table

#Now, we will use 'for' loop and check for different values of mtry.
a=c()
i=5
for (i in 3:8) {
  RF_M3 <- randomForest(Attrition ~ ., data = Test, ntree = 500, mtry = i, importance = TRUE)
  RF_Test_Pred1 <- predict(RF_M3, Test, type = "class")
  a[i-2] = mean(RF_Test_Pred1 == Test$Attrition)
}

a
windows()
plot(3:8,a)

