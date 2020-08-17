
#安裝tensorflow函式庫
install.packages('tensorflow')
#安裝keras函式庫
install.packages('keras')

#使用GPU運算
library(keras)
install_keras(tensorflow = "gpu")

#devtools::install_github("rstudio/keras") 
#library(keras)
#install_keras(tensorflow = "gpu")
#library(keras)
#install_keras(tensorflow = "gpu")
# Test Keras: Define a simple DNN network
require(keras)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>% 
  layer_dropout(rate = 0.6) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)
sessionInfo()

X  <-  as.matrix(read.csv("C:/Users/RenWu/Desktop/04_遞歸神經網路/資料集/20180924fish_x.csv"))
Y  <-  as.matrix(read.csv("C:/Users/RenWu/Desktop/04_遞歸神經網路/資料集/20180924fish_y.csv"))

#正規化
X <- X / 40
Y <- Y / 40

#轉換為matrix資料型態
X <- data.matrix(X)
X <- array_reshape(X, c(nrow(X), 2, 4))
                                       
#設定亂數種子 , disable_gpu = FALSE, disable_parallel_cpu = TRUE,quiet = TRUE
use_session_with_seed(0)


#設定神經網路學習目標
model <- keras_model_sequential() 
model %>% 
  layer_simple_rnn(units = 3, activation = "linear", use_bias = TRUE, input_shape =c(2, 4)) %>% 
  layer_dense(units = 3, activation = "softmax") %>% 
  layer_dense(units = 1, activation = "linear")
model %>% compile(
  loss='mean_squared_error', #最小平方誤差
  optimizer='sgd', #梯度下降
  metrics = c("accuracy")
)


#訓練神經網路
history <- model %>% fit(
  X, #輸入參數
  Y, #輸出參數
  epochs = 250, #訓練回合數
  batch_size = 1 #逐筆修正權重
)

#顯示神經網路權重值
model$get_weights().

#test renwu
test_X  <-  as.matrix(read.csv("C:/Users/RenWu/Desktop/04_遞歸神經網路/資料集/20180930fish_x.csv"))

#正規化
test_X <- test_X / 40
test_Y <- test_Y / 40

#轉換為matrix資料型態
test_X <- data.matrix(test_X)
test_X <- array_reshape(test_X, c(nrow(test_X), 2, 1))
#test renwu

#將測試資料代入模型進行預測,並取得預測結果
results <- model$predict(test_X)


#呈現估計結果
print(results)
