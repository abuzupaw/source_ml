library(neuralnet)
library(Metrics)
library(FNN)


myData <- read.csv("C://Users/acer/Documents/kuliah_nn_mlr.csv", sep=",")

jmlBaris <- NROW(myData) # untuk mengetahui jlm baris


dataPelatihan <- myData[1:(jmlBaris - 2),]

dataUji <- myData[(jmlBaris-1) :jmlBaris,]

#fungei neural network
nnmodel <-  neuralnet(UAS ~ IQ + Absen,data = dataPelatihan,hidden = c(2,2), linear.output = T)

plot(nnmodel)


mlmmodel <-
  lm(
    UAS ~ IQ + Absen,
    data = dataPelatihan,
  )

hasilmlm <- predict.lm(mlmmodel, dataUji[,1:2])

hasilnnbiasa <- compute(nnmodel, dataUji[,1:2])


errormlm = rmse(dataUji$UAS, hasilmlm)

errornn = rmse(dataUji$UAS, hasilnnbiasa$net.result) 


pred_k_1 = FNN::knn.reg(train = dataPelatihan[,1:2], test = dataUji[,1:2], y = dataPelatihan[,3], k = 1)
pred_k_2 = FNN::knn.reg(train = dataPelatihan[,1:2], test = dataUji[,1:2], y = dataPelatihan[,3], k = 2)
pred_k_3 = FNN::knn.reg(train = dataPelatihan[,1:2], test = dataUji[,1:2], y = dataPelatihan[,3], k = 3)


error_knn_k_1  = rmse(dataUji$UAS, pred_k_1$pred)
error_knn_k_2  = rmse(dataUji$UAS, pred_k_2$pred)
error_knn_k_3  = rmse(dataUji$UAS, pred_k_3$pred)




####################################################################################################
############################################################################################


# maxs <- apply(myData, 2, max) 
# mins <- apply(myData, 2, min)
# scaled <- as.data.frame(scale(myData, center = mins, scale = maxs - mins))
# 
# dataPelatihan <- scaled[1:(jmlBaris - 2),]
# 
# dataUji <- scaled[(jmlBaris-1) :jmlBaris,]
# 
# nnmodel <-
#   neuralnet(
#     UAS ~ IQ + Absen,
#     data = dataPelatihan,
#     hidden = c(2), linear.output = T)
