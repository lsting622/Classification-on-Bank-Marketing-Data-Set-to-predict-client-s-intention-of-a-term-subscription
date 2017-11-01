setwd("/Users/shutingliao/Documents/2017\ Winter/stats\ 503/data\ set/bank/")
require(ggplot2)
require(reshape2)
require(gridExtra)
library(ROCR)
bank=read.table('bank-full.csv', sep = ';', header = T)
y = as.factor(bank$y)
#bank1 = model.matrix(y~., bank)
#bank1 = data.frame(cbind(bank1,bank$y))
#bank1 = bank1[,-c(1)]
#bank_cont = bank1[,c(1,19,24,36,37)]



## set training data and test data
bank_yes = subset(bank, bank$y == "yes")
index <- sample(rownames(bank_yes), 0.2*nrow(bank_yes))
index = as.numeric(index)
test_bank_yes = bank[index, ]
bank_no = subset(bank, bank$y == "no")
index_no <- sample(rownames(bank_no), 0.2*nrow(bank_no))
index_no = as.numeric(index_no)
test_bank_no = bank[index_no, ]
index_total = c(index,index_no)
train_bank  = bank[-index_total,]
test_bank = rbind(test_bank_yes, test_bank_no)
test_bank = bank[index_total,]

## resampling
library(DMwR)
train_bank$y= as.factor(train_bank$y)
train_bank_factor = SMOTE(y~.,train_bank, k=5, perc.over = 100,perc.under = 200)
summary(train_bank_factor)

##set binary variable
test_bank_binary = model.matrix(y~., test_bank)
test_bank_binary = data.frame(cbind(test_bank_binary,test_bank$y))
test_bank_binary = test_bank_binary[,-c(1)]
train_bank_binary = model.matrix(y~., train_bank_factor)
train_bank_binary = data.frame(cbind(train_bank_binary,train_bank_factor$y))
train_bank_binary = train_bank_binary[,-c(1)]

## get smaller sample for easier boundary plot
smp_size <- floor(0.2 * nrow(train_bank_factor))
set.seed(48105)
sample.index <- sample(seq_len(nrow(train_bank_factor)), size = smp_size)
sample_train <- train_bank_factor[sample.index, ]
sample_train_fac <- na.omit(sample_train)
sample_duration_fac = sample_train_fac$duration
sample_pdays_fac = sample_train_fac$pdays
sample_train_bin = model.matrix(y~., sample_train_fac)
sample_train_bin = data.frame(cbind(sample_train_bin,sample_train_fac$y))
sample_train_bin= sample_train_bin[,-c(1)]
sample_pdays_bin = sample_train_bin$pdays
sample_duration_bin = sample_train_bin$duration

library(randomForest)
bankrf = randomForest(y~., mtry = 5, ntree = 2000, data=train_bank_factor, sampsize = c(1000,1000))
  ## error rate
bankrf_pred = predict(bankrf, test_bank)
mean(bankrf_pred != test_bank$y)
mean(bankrf_pred[test_bank$y=='no']=='yes')
mean(bankrf_pred[test_bank$y=='yes']=='no')
  ## using AUC
pred_rf = prediction(as.numeric(bankrf_pred), as.numeric(test_bank$y))
perf_rf = performance(pred_rf,'auc' )
as.numeric(perf_rf@y.values)
perf_rf_roc = performance(pred_rf,"tpr","fpr")
plot(perf_rf_roc, col="blue",lty=3, lwd=3, cex.lab=1.5,cex.axis=2,
     cex.main=1.5, main="ROC plot for Random Forest")
  ## boundary plot
project_fac = data.frame(cbind(sample_duration_fac,sample_pdays_fac,sample_train_fac$y))
names(project_fac) = c("duration", "pdays", "Class")
v1 = seq(min(project_fac[,1]), max(project_fac[,1]), length=500)
v2 = seq(min(project_fac[,2]), max(project_fac[,2]), length=500)
grid = expand.grid(duration = v1, pdays = v2)
class_train_rf = randomForest(Class~., mtry = 5, ntree = 2000, data=project_fac)
grid$class = predict(class_train_rf, grid) 
ggplot(data=project_fac, aes(x=duration, y=pdays, color=as.factor(Class))) +
  geom_contour(data=grid, aes(z=as.numeric(class)),
               color="black",size=0.5)+
  geom_point(size=0.5,aes(color=as.factor(Class)))+ggtitle('Random Forest')+
  scale_color_discrete(name='class',
                       labels=c('no', 'yes'))



#boost
library(adabag)
bank_boost = boosting(y~., data = train_bank_factor, mfinal = 100)
  ##error rate
boost_pred = predict.boosting(bank_boost, newdata = test_bank)
boost_pred$confusion
mean(boost_pred$class != test_bank$y)
mean(boost_pred$class[test_bank$y=='no']=='yes')
mean(boost_pred$class[test_bank$y=='yes']=='no')


  ##comparing error evolution in training and test set
errorevol(bank_boost,newdata=train_bank_factor)->evol.train_boost
errorevol(bank_boost,newdata=test_bank)->evol.test_boost
plot.errorevol(evol.test_boost,evol.train_boost)

  ##AUC
boost_pred2 = ifelse(boost_pred$class == 'yes',2,1)
pred_boost = prediction(boost_pred2, as.numeric(test_bank$y))
perf_boost = performance(pred_boost,'auc' )
as.numeric(perf_boost@y.values)
perf_boo_roc = performance(pred_boost,"tpr","fpr")
plot(perf_boo_roc, col="blue",lty=3, lwd=3, cex.lab=1.5,cex.axis=2,cex.main=1.5, main="ROC plot")

 ##boundary plot
class_train_boo = boosting(Class~., data=project_fac)
grid$class = predict(class_train_boo, grid) 
ggplot(data=project_fac, aes(x=duration, y=pdays, color=as.factor(Class))) +
  geom_contour(data=grid, aes(z=as.numeric(class)),
               color="black",size=0.5)+
  geom_point(size=0.5,aes(color=as.factor(Class)))+ggtitle('boosting')+
  scale_color_discrete(name='class',
                       labels=c('no', 'yes'))


#bag
bank_bag = bagging(y~., data = train_bank_factor, mfinal = 100,
                   control=rpart.control(maxdepth=5, minsplit=15))
##error rate
bank.bagging.pred <- predict.bagging(bank_bag,newdata=test_bank, newmfinal=50)
bank.bagging.pred$confusion
bank.bagging.pred$error
mean(bank.bagging.pred$class[test_bank$y=='no']=='yes')
mean(bank.bagging.pred$class[test_bank$y=='yes']=='no')

##comparing error evolution in training and test set
errorevol(bank_bag,newdata=train_bank_factor)->evol.train
errorevol(bank_bag,newdata=train_bank)->evol.test
plot.errorevol(evol.test,evol.train)

##AUC
bag_pred2 = ifelse(bank.bagging.pred$class == 'yes', 2, 1)
pred_bag = prediction(bag_pred2, as.numeric(test_bank$y))
perf_bag = performance(pred_bag,'auc' )
as.numeric(perf_bag@y.values)
perf_bag_roc = performance(pred_bag,"tpr","fpr")
plot(perf_bag_roc, col="blue",lty=3, lwd=3, cex.lab=1.5,cex.axis=2,cex.main=1.5, main="ROC plot")

 ##plot
class_train_bag = bagging(Class~., data=project_fac)
grid$class = predict(class_train_bag, grid) 
ggplot(data=project_fac, aes(x=duration, y=pdays, color=as.factor(Class))) +
  geom_contour(data=grid, aes(z=as.numeric(class)),
               color="black",size=0.5)+
  geom_point(size=0.5,aes(color=as.factor(Class)))+ggtitle('bagging')+
  scale_color_discrete(name='class',
                       labels=c('no', 'yes'))


#tree
library(rpart)
bank_tr = rpart(train_bank_factor$y~., data=train_bank_factor)
bank_tr
library(rpart.plot)
rpart.plot(bank_tr, type = 4,extra = 1,clip.right.labs = F)

control = rpart.control(cp = 0.05, minsplit = 2)
bank_tr2 = rpart(train_bank_factor$y~., data=train_bank_factor,
                 parms = list(split = "gini"), control = control)
rpart.plot(bank_tr2, type = 4, extra = 1,clip.right.labs = F)

control = rpart.control(cp = 0.000, xxval = 100, minsplit = 2)
bank_tr3 = rpart(y~., data=train_bank_factor, control = control)
plotcp(bank_tr3)
printcp(bank_tr3)
selected_tr <- prune(bank_tr, cp= bank_tr$cptable[which.min(bank_tr$cptable[,"xerror"]),"CP"])
plotcp(selected_tr) #much worse
##error
banktr_pred = predict(bank_tr, test_bank, type = "class") 
mean(banktr_pred != test_bank$y)
mean(banktr_pred[test_bank$y=='no']=='yes')
mean(banktr_pred[test_bank$y=='yes']=='no')
## using AUC
pred_tr = prediction(as.numeric(banktr_pred), as.numeric(test_bank$y))
perf_tr = performance(pred_tr,'auc' )
as.numeric(perf_tr@y.values)
perf_tr_roc = performance(pred_tr,"tpr","fpr")
plot(perf_tr_roc, col="blue",lty=3, lwd=3, cex.lab=1.5,cex.axis=2,cex.main=1.5, main="ROC plot")
 ##boundary plot
class_train_tr = rpart(Class~., data=project_fac)
grid$class = predict(class_train_tr, grid) 
ggplot(data=project_fac, aes(x=duration, y=pdays, color=as.factor(Class))) +
  geom_contour(data=grid, aes(z=as.numeric(class)),
               color="black",size=0.5)+
  geom_point(size=0.5,aes(color=as.factor(Class)))+ggtitle('Tree')+
  scale_color_discrete(name='class',
                       labels=c('no', 'yes'))


#SVM
#choose parameter and kernel
#AUC
##Test Auc
test_auc_svm = function(costC, ker = "radial",...) {
  #Train
  bank.svm = svm(V44 ~ ., data=train_bank_binary, kernel= ker, cost=costC,...)
  bank_svm_pred = predict(bank.svm, test_bank_binary)
  pred = prediction(as.numeric(bank_svm_pred), as.numeric(test_bank_binary$V44))
  perf = performance(pred,'auc' )
  test_auc_val = as.numeric(perf@y.values)
  return(test_auc_val)
}

##Train Auc
train_auc_svm = function(costC, ker = "radial",...) {
  #Train
  bank.svm = svm(V44 ~ ., data=train_bank_binary, kernel= ker, cost=costC,...)
  pred = prediction(as.numeric(bank.svm$fitted), as.numeric(train_bank_binary$V44))
  perf = performance(pred,'auc' )
  train_auc_val = as.numeric(perf@y.values)
  return(train_auc_val)
}

#Train auc
costs = c(0.1,1,10,100)
train_auc_l = sapply(costs, function(cost) train_auc_svm(cost, 'linear'))
train_auc_errsl = data.frame(train_auc_l)
train_auc_gau = sapply(costs, function(cost) train_auc_svm(cost, 'radial'))
train_auc_errsg = data.frame(train_auc_gau)
train_auc_poly = sapply(costs, function(cost) train_auc_svm(cost, 'polynomial'))
train_auc_errsp = data.frame(train_auc_poly)
train_auc_errs1 = data.frame(train_auc_errsl, train_auc_errsg, train_auc_errsp, costs)
colnames(train_auc_errs1) = c("Linear", "Gaussian", "Polynomial","cost")
#test auc
costs = c(0.1,1,10,100)
test_auc_l = sapply(costs, function(cost) test_auc_svm(cost, 'linear'))
test_auc_errsl = data.frame(test_auc_l)
test_auc_gau = sapply(costs, function(cost) test_auc_svm(cost, 'radial'))
test_auc_errsg = data.frame(test_auc_gau)
test_auc_poly = sapply(costs, function(cost) test_auc_svm(cost, 'polynomial'))
test_auc_errsp = data.frame(test_auc_poly)
test_auc_errs1 = data.frame(test_auc_errsl, test_auc_errsg, test_auc_errsp, costs)
colnames(test_auc_errs1) = c("Linear", "Gaussian", "Polynomial","cost")

d1_auc = melt(train_auc_errs1, id="cost")
ggplot(d1_auc, aes_string(x="cost", y="value", colour="variable", linetype="variable")) + 
  geom_line(size=1) + labs(x = "Cost",
                           y = "Area Under Curve",
                           colour="",group="",
                           linetype="",shape="") + ggtitle("SVM Kernels Train") + scale_x_log10()

d2_auc = melt(test_auc_errs1, id="cost")
ggplot(d2_auc, aes_string(x="cost", y="value", colour="variable", linetype="variable")) + 
  geom_line(size=1) + labs(x = "Cost",
                           y = "Area Under Curve",
                           colour="",group="",
                           linetype="",shape="") + ggtitle("SVM Kernels Test") + scale_x_log10()




train_cv_auc = function(train_bank_binary, test_bank_binary, costC) {
  #Gaussian gamma test
  svm_1 = svm(V44 ~ ., data=train_bank_binary, type='C-classification',
              kernel="radial", cost=costC, gamma=0.001)
  pred_1=predict(svm_1, test_bank_binary)
  pred_g1 = prediction(as.numeric(pred_1),as.numeric(test_bank_binary$V44))
  perf_g1 = performance(pred_g1,'auc' )
  auc_g1 = as.numeric(perf_g1@y.values)
  
  svm_2 =svm(V44 ~ ., data=train_bank_binary, type='C-classification',
             kernel="radial", cost=costC, gamma=0.01)
  pred_2=predict(svm_2, test_bank_binary)
  pred_g2 = prediction(as.numeric(pred_2),as.numeric(test_bank_binary$V44))
  perf_g2 = performance(pred_g2,'auc' )
  auc_g2 = as.numeric(perf_g2@y.values)
  
  svm_3 =svm(V44 ~ ., data=train_bank_binary, type='C-classification',
             kernel="radial", cost=costC, gamma=0.05)
  pred_3=predict(svm_3, test_bank_binary)
  pred_g3 = prediction(as.numeric(pred_3),as.numeric(test_bank_binary$V44))
  perf_g3 = performance(pred_g3,'auc' )
  auc_g3 = as.numeric(perf_g3@y.values)
  
  svm_4=svm(V44 ~ ., data=train_bank_binary, type='C-classification',
            kernel="radial", cost=costC, gamma=0.1)
  pred_4=predict(svm_4, test_bank_binary)
  pred_g4 = prediction(as.numeric(pred_4),as.numeric(test_bank_binary$V44))
  perf_g4 = performance(pred_g4,'auc' )
  auc_g4 = as.numeric(perf_g4@y.values)
  
  svm_5=svm(V44 ~ ., data=train_bank_binary, type='C-classification',
            kernel="radial", cost=costC, gamma=2)
  pred_5=predict(svm_5, test_bank_binary)
  pred_g5 = prediction(as.numeric(pred_5),as.numeric(test_bank_binary$V44))
  perf_g5 = performance(pred_g5,'auc' )
  auc_g5 = as.numeric(perf_g5@y.values)
  return(c(auc_g1,auc_g2,auc_g3,auc_g4,auc_g5))
}

k_aucs = sapply(c(0.01,1,10,100,1000), function(c) train_cv_auc(as.data.frame(train_bank_binary), as.data.frame(test_bank_binary),c))
df_aucs = data.frame(t(k_aucs), c(0.01,1,10,100,1000))
colnames(df_aucs) = c('gamma0.001',  'gamma0.01', 'gamma0.05','gamma0.1','gamma1', 'Cost')

dataL_auc <- melt(df_aucs, id="Cost")
ggplot(dataL_auc, aes_string(x="Cost", y="value", colour="variable",
                             group="variable", linetype="variable", shape="variable")) +
  geom_line(size=1) + labs(x = "Number of Parameter Cost",
                           y = "Area Under Curve",
                           colour="",group="",
                           linetype="",shape="") + 
  geom_point(size=2) + scale_x_log10() + ggtitle("Gaussian Test")

#use the parameter we have chosen
library(e1071)
project_svm = data.frame(cbind(sample_duration_bin,sample_pdays_bin,sample_train_bin$V44))
names(project_svm) = c("duration", "pdays", "Class")
class_train_svm = svm(Class~., data=project_svm, type="C-classification",
                     kernel ="radial", cost=10, gamma = 0.01)
v1 = seq(min(project_svm[,1]), max(project_svm[,1]), length=500)
v2 = seq(min(project_svm[,2]), max(project_svm[,2]), length=500)
grid = expand.grid(duration = v1, pdays = v2)
grid$class = predict(class_train_svm, grid) 
ggplot(data=project_svm, aes(x=duration, y=pdays, color=as.factor(Class))) +
  geom_contour(data=grid, aes(z=as.numeric(class)),
               color="black",size=0.5)+
  geom_point(size=0.5,aes(color=as.factor(Class)))+ggtitle('SVM')+
  scale_color_discrete(name='class',
                       labels=c('no', 'yes'))
                       #labels=c('G', 'M', 'B'))



#LR
library(MASS)
logit.fit = glm(V44~., data = train_bank_binary)
  ## error rate
glm.prob=predict(logit.fit,test_bank_binary,type="response")
glm.prob = 1+(glm.prob>1.5)
table(glm.prob,test_bank_binary$V44)
mean(glm.prob != test_bank_binary$V44)
mean(glm.prob[test_bank_binary$V44=='1']=='2')
mean(glm.prob[test_bank_binary$V44=='2']=='1')
  ## using AUC
pred_lr = prediction(as.numeric(glm.prob), as.numeric(test_bank_binary$V44))
perf_lr = performance(pred_lr,'auc' )
as.numeric(perf_lr@y.values)
perf_lr_roc = performance(pred_lr,"tpr","fpr")
plot(perf_lr_roc, col="blue",lty=3, lwd=3, cex.lab=1.5,cex.axis=2,cex.main=1.5, main="ROC plot")
  ##boundary plot
project_lr = data.frame(cbind(sample_duration_bin,sample_pdays_bin,sample_train_bin$V44-1))
names(project_lr) = c("duration", "pdays", "Class")
class_train_lr = glm(Class~., data=project_lr,family = "binomial")
v1 = seq(min(project_lr[,1]), max(project_lr[,1]), length=500)
v2 = seq(min(project_lr[,2]), max(project_lr[,2]), length=500)
grid = expand.grid(duration = v1, pdays = v2)
grid$logit =1*(predict(class_train_lr, grid, type='response') >.5)
ggplot(data=project_lr, aes(x=duration, y=pdays, color=as.factor(Class))) +
  geom_contour(data=grid, aes(z=as.numeric(logit)),
               color="black",size=0.5)+
  geom_point(size=0.5,aes(color=as.factor(Class)))+ggtitle('Logistic Regression')+
  scale_color_discrete(name='class',
                       labels=c('no', 'yes'))
#geom_abline(slope = -5.134e-03/6.546e-03, intercept = 2.033e+00/6.546e-03, linetype = 2)+
#geom_abline(slope = -0.0053811/0.0056434, intercept = 2.0884664/0.0056434, linetype = 1)+
  
#auc plot for several models
##random Forest
library(randomForest)
bankrf = randomForest(y~., mtry = 5, ntree = 2000, data = train_bank_factor, replace=FALSE)
bankrf_pred = predict(bankrf, test_bank)
pred_rf = prediction(as.numeric(bankrf_pred), as.numeric(test_bank$y))
perf_roc_rf = performance(pred_rf, 'tpr', 'fpr')
perf_roc_rf_val = data.frame(perf_roc_rf@x.values, perf_roc_rf@y.values)
colnames(perf_roc_rf_val) = c('False_positive', 'True_positive')
perf_roc_rf_val$type = 'Random Forest'

##Tree
bank_tr = rpart(y~., data = train_bank_factor)
bank_tr_pred = predict(bank_tr, test_bank)
bank_tr_pred = ifelse(bank_tr_pred[,1]>bank_tr_pred[,2],1,2)
pred_tr = prediction(as.numeric(bank_tr_pred),as.numeric(test_bank$y))
perf_roc_tr = performance(pred_tr,'tpr', 'fpr')
perf_roc_tr_val = data.frame(perf_roc_tr@x.values, perf_roc_tr@y.values)
colnames(perf_roc_tr_val) = c('False_positive', 'True_positive')
perf_roc_tr_val$type = 'Classification Tree'

##bagging
bank.bagging = bagging(y~.,data = train_bank_factor,mfinal=15,control=rpart.control(maxdepth=5, minsplit=15))

bank_bag_pred = predict.bagging(bank.bagging,newdata=test_bank, newmfinal=10)
bank_bag_pred = ifelse(bank_bag_pred$class == 'no',1,2)
pred_bag = prediction(as.numeric(bank_bag_pred),as.numeric(test_bank$y))
perf_roc_bag = performance(pred_bag,'tpr', 'fpr')
perf_roc_bag_val = data.frame(perf_roc_bag@x.values,perf_roc_bag@y.values)
colnames(perf_roc_bag_val) = c('False_positive', 'True_positive')
perf_roc_bag_val$type = 'bagging'

##boosting
bank_boost = boosting(y~., data = train_bank_factor, mfinal = 100)
boost_pred = predict.boosting(bank_boost, newdata = test_bank, newmfinal = 100)
bank_boost_pred = ifelse(boost_pred$class == 'no',1,2)
pred_boost = prediction(as.numeric(bank_boost_pred),as.numeric(test_bank$y))
perf_roc_boost = performance(pred_boost,'tpr', 'fpr')
perf_roc_boost_val = data.frame(perf_roc_boost@x.values, perf_roc_boost@y.values)
colnames(perf_roc_boost_val) = c('False_positive', 'True_positive')
perf_roc_boost_val$type = 'boosting'

##SVM
bank_svm = svm(V44 ~ ., data=train_bank_binary, type='C-classification', kernel="radial", cost=10, gamma=0.01)
svm_pred = predict(bank_svm, newdata = test_bank_binary)
pred_svm = prediction(as.numeric(svm_pred),as.numeric(test_bank_binary$V44))
perf_roc_svm = performance(pred_svm,'tpr', 'fpr')
perf_roc_svm_val = data.frame(perf_roc_svm@x.values,perf_roc_svm@y.values)
colnames(perf_roc_svm_val) = c('False_positive', 'True_positive')
perf_roc_svm_val$type = 'SVM'

##plot
model_aucs = rbind(perf_roc_svm_val, perf_roc_boost_val)
model_aucs = rbind(model_aucs, perf_roc_bag_val)
model_aucs = rbind(model_aucs, perf_roc_tr_val)
model_aucs = rbind(model_aucs, perf_roc_rf_val)
ggplot(model_aucs, aes_string(x="False_positive", y="True_positive", colour="type",
                              group="type", linetype="type", shape="type")) +
  geom_line(size=1) + labs(x = "False positive rate",
                           y = "True positive rate",
                           colour="",group="",
                           linetype="",shape="") + 
  geom_point(size=2) + ggtitle("ROC")
