library(tidyverse)
library(quantmod)
library(tseries)
library(caret)
library(xts)
library(scales)

#To load keras and tensorflow
remove.packages(c("keras", "tensorflow", "reticulate")) 
install.packages("reticulate")
library(reticulate)

use_condaenv(condaenv = "base", conda = "D:/Anaconda/condabin/conda.bat")
conda_create("r_keras", python_version = "3.7")
use_condaenv("r_keras")

#Need to run again 
py_install("tensorflow")
py_install("keras")
py_install("numpy")

install.packages("tensorflow")
install.packages("keras")

library(tensorflow)
library(keras)

# Test tensorflow
tf$constant("Hello TensorFlow!") 



#Setting seeds to ensure we get the same results each time we run the code
set.seed(19010)                
tf$random$set_seed(19010)      

ticker <- "AMGN"                        
start_date <- as.Date("2010-01-01")     
end_date <- as.Date("2024-01-01")       

getSymbols(ticker, src = "yahoo", from = start_date, to = end_date)
data <- na.omit(AMGN[, "AMGN.Close"])

scale_data <- function(x) {
  scaled <- scale(x)
  
  scale_param <- attr(scaled, "scaled:scale")
  center_param <- attr(scaled, "scaled:center")
  
  attr(scaled, "scale") <- scale_param
  attr(scaled, "center") <- center_param
  
  return(scaled)
}

data_scaled <- scale_data(data)


create_sequences <- function(data_array, window_size) {
  
  n <- length(data_array)
  
  X <- matrix(0, nrow = n - window_size, ncol = window_size)
  y <- numeric(n - window_size)
  
  for(i in (window_size + 1):n) {
    X[i - window_size, ] <- data_array[(i - window_size):(i - 1)]
    y[i - window_size] <- data_array[i]
  }
  
  return(list(X = X, y = y))
}

window_size <- 60  #Using 60 days of historical data to predict the next day

sequences <- create_sequences(data_scaled, window_size)


# TRAIN-TEST SPLIT
# ============================================================================
split_ratio <- 0.8  
split_index <- floor(nrow(sequences$X) * split_ratio)

X_train <- sequences$X[1:split_index, ]
X_test <- sequences$X[(split_index + 1):nrow(sequences$X), ]
y_train <- sequences$y[1:split_index]
y_test <- sequences$y[(split_index + 1):length(sequences$y)]

#Reshaping data for LSTM/GRU
X_train <- array_reshape(X_train, c(nrow(X_train), window_size, 1))
X_test <- array_reshape(X_test, c(nrow(X_test), window_size, 1))


# MODEL BUILDING
# ============================================================================

build_lstm_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = 50,                    
      return_sequences = TRUE,        
      input_shape = input_shape       
    ) %>%
    layer_dropout(rate = 0.2) %>%
    layer_lstm(units = 50) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1)             
  
  model %>% compile(optimizer = "adam",    
    loss = "mse")
  
  return(model)
}

build_gru_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_gru(
      units = 50,
      return_sequences = TRUE,
      input_shape = input_shape
    ) %>%
    layer_dropout(rate = 0.2) %>%
    layer_gru(units = 50) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = "adam",
    loss = "mse")
  
  return(model)
}


# MODEL TRAINING SETUP
# ============================================================================

input_shape <- c(window_size, 1)  
epochs <- 50                       
batch_size <- 32                  

lstm_model <- build_lstm_model(input_shape)
gru_model <- build_gru_model(input_shape)


early_stopping <- callback_early_stopping(
  monitor = "val_loss",           
  patience = 10,                  
  restore_best_weights = TRUE)


# MODEL TRAINING
# ============================================================================

cat("Training LSTM model...\n")
history_lstm <- lstm_model %>% fit(
  X_train, y_train,
  epochs = epochs,
  batch_size = batch_size,
  validation_split = 0.1,         
  callbacks = list(early_stopping),
  verbose = 1
)


cat("Training GRU model...\n")
history_gru <- gru_model %>% fit(
  X_train, y_train,
  epochs = epochs,
  batch_size = batch_size,
  validation_split = 0.1,
  callbacks = list(early_stopping),
  verbose = 1
)


# MAKING PREDICTIONS
# ============================================================================
preds_lstm <- predict(lstm_model, X_test)
preds_gru <- predict(gru_model, X_test)

ensemble_preds <- (preds_lstm + preds_gru) / 2


# INVERSE SCALING
# ============================================================================
inverse_scale <- function(x, scaled_obj) {
  scale_param <- attr(scaled_obj, "scale")
  center_param <- attr(scaled_obj, "center")
  
  if (is.null(scale_param) | is.null(center_param)) {
    stop("Scaling parameters are missing. Ensure the data was properly scaled.")
  }
  
  return(x * scale_param + center_param)
}

y_test_original <- inverse_scale(y_test, data_scaled)
lstm_preds_original <- inverse_scale(preds_lstm, data_scaled)
gru_preds_original <- inverse_scale(preds_gru, data_scaled)
ensemble_preds_original <- inverse_scale(ensemble_preds, data_scaled)


# MODEL EVALUATION
# ============================================================================
mse <- function(actual, predicted) mean((actual - predicted)^2)
rmse <- function(actual, predicted) sqrt(mse(actual, predicted))
mae <- function(actual, predicted) mean(abs(actual - predicted))

#Calculating performance metrics for all models
metrics <- data.frame(
  Model = c("LSTM", "GRU", "Ensemble"),
  MSE = c(mse(y_test_original, lstm_preds_original),
    mse(y_test_original, gru_preds_original),
    mse(y_test_original, ensemble_preds_original)),
  RMSE = c(rmse(y_test_original, lstm_preds_original),
    rmse(y_test_original, gru_preds_original),
    rmse(y_test_original, ensemble_preds_original)),
  MAE = c(mae(y_test_original, lstm_preds_original),
    mae(y_test_original, gru_preds_original),
    mae(y_test_original, ensemble_preds_original)))

print("Model Performance Comparison:")
print(metrics)

# VISUALIZATION
# ============================================================================
test_dates <- index(data)[(length(data) - length(y_test) + 1):length(data)]

plot_data <- data.frame(
  Date = test_dates,
  Actual = y_test_original,
  LSTM = as.vector(lstm_preds_original),
  GRU = as.vector(gru_preds_original),
  Ensemble = as.vector(ensemble_preds_original)
)
str(plot_data)

ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
  geom_line(aes(y = LSTM, color = "LSTM"), alpha = 0.7) +
  geom_line(aes(y = GRU, color = "GRU"), alpha = 0.7) +
  geom_line(aes(y = Ensemble, color = "Ensemble"), alpha = 0.7) +
  scale_color_manual(values = c(
    "Actual" = "black",
    "LSTM" = "blue",
    "GRU" = "green",
    "Ensemble" = "red")) +
  labs(title = paste(ticker, "Stock Price Prediction - Model Comparison"),
    x = "Date",
    y = "Stock Price",
    color = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position ="bottom")
