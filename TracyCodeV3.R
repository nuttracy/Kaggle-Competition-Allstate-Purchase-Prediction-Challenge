### Kaggle Competition: Allstate Purchase Prediction Challenge
### Code by Lezhi Tracy Wang
### Using trainning sets to validate model.
#library(car)
#library(ggplot2)
library(randomForest)

rm(list = ls())

#
tune = T

#read whole data
wholeData <- read.csv("train.csv")
UserID <- wholeData$customer_ID
uniID <- unique(UserID) #unique plan
lenID <- length(uniID)
smp_size <- floor(0.7*lenID) #70% training data, 30% test data
set.seed(123)
trainInd <- sample(lenID, size = smp_size)

trainID <- sort(uniID[trainInd])
#testID <- sort(uniID[-trainInd]) 
cmpID <- UserID %in% trainID
selInd <- which(cmpID==TRUE)
trainData <- wholeData[selInd, ]

ind <- trainData$record_type==1 #find shopping record
dsub0 <- trainData[c(ind[3:length(ind)],FALSE,FALSE),] #find the 2nd last viewed options before purchased
dsub1 <- trainData[c(ind[2:length(ind)],FALSE),] #find the last viewed options before purchased
dsub2 <- trainData[ind,] #find the purchased options

plantrain <- paste0(trainData$A, trainData$B, trainData$C, trainData$D, trainData$E, trainData$F, trainData$G)
planv <- paste0(dsub1$A, dsub1$B, dsub1$C, dsub1$D, dsub1$E, dsub1$F, dsub1$G)
planp <- paste0(dsub2$A, dsub2$B, dsub2$C, dsub2$D, dsub2$E, dsub2$F, dsub2$G)

uniplan <- unique(plantrain)
length(uniplan) #unique plans
cmpplan <- ifelse(planv == planp, TRUE, FALSE)
res1 = table(cmpplan)["TRUE"]
pres1 = as.numeric(res1)/length(planp) #percentage of last viewed options = purchased options
print(pres1)

dsub1$day<-ifelse(dsub1$day<=4,1,0) #date
dsub1$time <- as.character(dsub1$time)
dsub1$time <- sprintf("%05s", dsub1$time)
dsub1$time <- as.integer(substr(dsub1$time, 1,2))*60+as.integer(substr(dsub1$time,4,5)) #express time by minutes

#manually reduce the number of levels in state to 32.
staterank <- order(table(dsub1$state), decreasing =T) 
levels(dsub1$state) <- c(levels(dsub1$state), "ZZ")
ind <- levels(dsub1$state)[staterank][32:36]
for (i in 1:length(ind))
{
  dsub1$state[dsub1$state == ind[i]] <- "ZZ"
}
dsub1$state <- factor(dsub1$state)

#modify NA with values
dsub1$risk_factor[is.na(dsub1$risk_factor)]<-3
dsub1$C_previous[is.na(dsub1$C_previous)]<-2
dsub1$duration_previous[is.na(dsub1$duration_previous)]<-6
dsub1$state<-factor(dsub1$state)

#combine the 2nd last viewed options and last viewed options
dsub1$AA <- paste0(dsub1$A, dsub0$A)
dsub1$BB <- paste0(dsub1$B, dsub0$B)
dsub1$CC <- paste0(dsub1$C, dsub0$C)
dsub1$DD <- paste0(dsub1$D, dsub0$D)
dsub1$EE <- paste0(dsub1$E, dsub0$E)
dsub1$FF <- paste0(dsub1$F, dsub0$F)
dsub1$GG <- paste0(dsub1$G, dsub0$G)

#predict options in trainning sets
dsub1$PA <- as.factor(dsub2$A)
dsub1$PB <- as.factor(dsub2$B)
dsub1$PC <- as.factor(dsub2$C)
dsub1$PD <- as.factor(dsub2$D)
dsub1$PE <- as.factor(dsub2$E)
dsub1$PF <- as.factor(dsub2$F)
dsub1$PG <- as.factor(dsub2$G)

for (i in length(dsub1))
{
  dsub1[, i] <- as.factor(dsub1[,i])
}

dtrain <- dsub1

##############
#read test data
#testData1 <- read.csv("test_v2.csv")

testData1 <- wholeData[-selInd, ]
#delete shopping results
inds1 <- testData1$record_type==1 #find shopping record
inds <- which(inds1 == TRUE)
testData <-testData1[-inds,]

ind <-which(!duplicated(testData$customer_ID, fromLast = TRUE))-1
dtsub0 <- testData[ind, ] #find the 2nd last viewed options before purchased
dtsub1 <- testData[ind+1,] #find the last viewed options before purchased

dtsub1$day<-ifelse(dtsub1$day<=4,1,0) #date
dtsub1$time <- as.character(dtsub1$time)
dtsub1$time <- sprintf("%05s", dtsub1$time)
dtsub1$time <- as.integer(substr(dtsub1$time, 1,2))*60+as.integer(substr(dtsub1$time,4,5)) #express time into minutes

#manually reduce the number of levels in state to 32.
staterank <- order(table(dtsub1$state), decreasing =T) 
levels(dtsub1$state) <- c(levels(dtsub1$state), "ZZ")
ind <- levels(dtsub1$state)[staterank][32:36]
for (i in 1:length(ind))
{
  dtsub1$state[dtsub1$state == ind[i]] <- "ZZ"
}
dtsub1$state <- factor(dtsub1$state)

#modify NA with values
dtsub1$risk_factor[is.na(dtsub1$risk_factor)]<-3
dtsub1$C_previous[is.na(dtsub1$C_previous)]<-2
dtsub1$duration_previous[is.na(dtsub1$duration_previous)]<-6

#modify NA locations
tmp <- split(testData$location, testData$state)
tmp2 <- sapply(tmp, function(x) x[1])
NAstate <- dtsub1[is.na(testData$location), "state"]
NAloc <- tmp2[NAstate]
testData$location[is.na(testData$location)]<-NAloc

#combine the 2nd last viewed options and last viewed options
dtsub1$AA <- paste0(dtsub1$A, dtsub0$A)
dtsub1$BB <- paste0(dtsub1$B, dtsub0$B)
dtsub1$CC <- paste0(dtsub1$C, dtsub0$C)
dtsub1$DD <- paste0(dtsub1$D, dtsub0$D)
dtsub1$EE <- paste0(dtsub1$E, dtsub0$E)
dtsub1$FF <- paste0(dtsub1$F, dtsub0$F)
dtsub1$GG <- paste0(dtsub1$G, dtsub0$G)

#modify NA with values
# testna <- which(is.na(trainData$C_previous)==F)
# pm <- mean(trainData$C_previous[testna])
# pm

dtesttmp <-dtsub1
dtesttmp$isTest <- rep(1, nrow(dtesttmp))
dtraintmp <- dtrain[,1:length(dtsub1)]
dtraintmp$isTest<- rep(0, nrow(dtraintmp))

dfullSet <- rbind(dtesttmp, dtraintmp)

for (i in length(dfullSet))
{
  dfullSet[,i] <-as.factor(dfullSet[,i])
}

dtesttmp <- dfullSet[dfullSet$isTest==1,]
dtest <- dtesttmp[,1:length(dtsub1)]
dtraintmp <- dfullSet[dfullSet$isTest==0,]
dtrain[,1:length(dtsub1)] <- dtraintmp[,1:length(dtsub1)]

##############
# model
mlist <- list()
dresult <- data.frame(customer_ID = dtest$customer_ID)

predItem <- c("PA", "PB", "PC", "PD", "PE", "PF", "PG")
featureall <- names(dtrain)
trainItem <- c("shopping_pt", "day", "time", "state", "group_size", "homeowner", 
              "car_age", "car_value", "risk_factor", "age_oldest", "age_youngest",
              "married_couple", "C_previous", "duration_previous", "A", "B", "C",
              "D", "E", "F", "G", "cost", "AA", "BB", "CC", "DD", "EE", "FF", "GG")

if (tune)
{
  cmtry <-rep(0, length(predItem))
  for (i in 1: length(predItem))
  {
    print(predItem[i])
    model.tune <- tuneRF(dtrain[, trainItem], dtrain[, predItem[i]])
    cmtry[i] <- model.tune[order(model.tune[,2])[1],1]
  }
  print(cmtry)
}

for (i in 1:length(predItem))
{
  print(predItem[i])
  if(tune)
  {
    model<-randomForest(dtrain[, trainItem], dtrain[, predItem[i]], mtry = cmtry[i])
    
  }
  else 
  {
    model<-randomForest(dtrain[,trainItem],dtrain[,predItem[i]])
  }
  mlist <-c(mlist, list(model))
  print(model$importance[order(-model$importance)[1:10], ])
  
  figtitle <- sprintf("Predict3 Option %s",LETTERS[i])
  figstr <- sprintf("P3%s.png",LETTERS[i])
  
  png(filename=figstr)
  varImpPlot(model,main = figtitle)
  dev.off()
  
}



for (i in 1:length(predItem))
{
  model <- mlist[[i]]
  pred <- predict(model,dtest[,trainItem])
  dresult[,predItem[i]]<-pred
}


dresult$plan <-paste0(dresult[, "PA"], dresult[,"PB"], dresult[,"PC"],dresult[,"PD"],dresult[,"PE"],dresult[,"PF"],dresult[,"PG"])

tresult <- testData1[inds,]
realresult <- data.frame(customer_ID = dtest$customer_ID)

realresult$plan <-tresult$plan <-paste0(tresult[, "A"], tresult[,"B"], tresult[,"C"],tresult[,"D"],tresult[,"E"],tresult[,"F"],tresult[,"G"])
trainresult <- data.frame(customer_ID = dresult$customer_ID, plan = dresult$plan)

cmpresult <- ifelse(realresult$plan == trainresult$plan, TRUE, FALSE)

res2 = table(cmpresult)["TRUE"]
pres2 = as.numeric(res2)/length(cmpresult) #percentage of last viewed options = purchased options
print("Accuracy")
print(pres2)

write.csv(dresult[, c("customer_ID", "plan")], paste0("resultTracyV3.txt"), quote= FALSE, row.names = FALSE)

## prediction
## dependencies between options(only in purchased options)

# tmpopt1 <- as.integer(dsub2$A)
# tmpopt2 <- as.integer(dsub2$B)
# tmpopt3 <- as.integer(dsub2$C)
# tmpopt4 <- as.integer(dsub2$D)
# tmpopt5 <- as.integer(dsub2$E)
# tmpopt6 <- as.integer(dsub2$F)
# tmpopt7 <- as.integer(dsub2$G)
# 
# tmpoptMat = cbind(tmpopt1,tmpopt2,tmpopt3,tmpopt4,tmpopt5,tmpopt6,tmpopt7)
# 
# for (i in 1:6)
# {
#   tmp1 = tmpoptMat[,i]
#   utmp1 = sort(unique(tmp1))
#   len1 = length(utmp1)
#   
#   for (j in (i+1):7)
#   {
#     tmp2 = tmpoptMat[,j]
#     str1 = sprintf("Option%s", LETTERS[i])
#     str2 = sprintf("Option%s", LETTERS[j])
#     dsubtmp <-dsub2[, c((i+17),(j+17))]
#     names(dsubtmp) <-c("Opt1","Opt2")
#     T1_names <-list()
# 
#     for(iname in 1:len1)
#     {
#       name <- sprintf('%s=%d',LETTERS[i],utmp1[iname])
#       tmp <- list(1:len1)
#       T1_names[iname] <- name
#     }
# 
#     T1_labeller <-function(variable, value){return(T1_names[value])}
#     plot1 <-ggplot(dsubtmp, aes(Opt2)) + geom_bar()+facet_grid(.~Opt1,labeller=as_labeller(T1_labeller))+
#             labs(x=str2, y = "FrequenpredItem")+ theme(text = element_text(size=18))
#     print(plot1)
#   }
# }

##
