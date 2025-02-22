library(tidyverse)
library(lubridate)
library(xgboost)
library(caret)
library(quantmod)
library(rsample)

rm(list = ls())

stock <- "AMGN"
data <- getSymbols.yahoo(stock, from = "2014-01-01", to = "2024-12-31",
                         periodicity = "daily", auto.assign = FALSE)[,4]
chartSeries(data)
data <- data.frame(Date = index(data), Close = as.numeric(data$AMGN.Close))

#Creating lagged values
data <- data %>%
  arrange(Date) %>%
  mutate(Close_Lag1 = lag(Close, 1),
         Close_Lag2 = lag(Close, 2),
         Target = log(Close / Close_Lag1)) %>%
  drop_na()

#Creating test and train
set.seed(18990)
split_index <- floor(0.8 * nrow(data))
train_data <- data[1:split_index, ]
test_data <- data[(split_index + 1):nrow(data), ]

# Create matrices
train_x <- as.matrix(train_data[, c("Close_Lag1", "Close_Lag2")])
train_y <- train_data$Target
test_x <- as.matrix(test_data[, c("Close_Lag1", "Close_Lag2")])
test_y <- test_data$Target

# Convert to XGBoost DMatrix
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

params <- list(
  objective = "reg:squarederror",
  booster = "gbtree",
  eval_metric = "rmse",
  eta = 0.1,  # Learning rate
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train model
set.seed(18990)
xgb_model <- xgb.train(params = params,
                       data = dtrain,
                       nrounds = 100,
                       watchlist = list(train = dtrain, test = dtest),
                       early_stopping_rounds = 10,
                       verbose = 1)

#Evaluation
pred_log <- predict(xgb_model, dtest)
pred_close <- test_data$Close_Lag1 * exp(pred_log)

rmse <- sqrt(mean((pred_close - test_data$Close)^2))
cat("Test RMSE:", rmse, "\n")

#Plot
plot_data <- data.frame(
  Date = test_data$Date,  
  Actual_Close = test_data$Close,  
  Predicted_Close = pred_close)

ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual_Close, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted_Close, color = "Predicted"), size = 1, linetype = "dashed") +
  labs(title = "Actual vs. Predicted Closing Prices",
       x = "Date",
       y = "Closing Price") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

importance <- xgb.importance(feature_names = colnames(train_x), model = xgb_model)
print(importance)
xgb.plot.importance(importance)


#Using rolling window
cv_folds <- rolling_origin(data, initial = 200, assess = 1, cumulative = FALSE)

cv_rmse <- c()

for (i in seq_along(cv_folds$splits)) {
  
  train_data_rw <- analysis(cv_folds$splits[[i]])
  test_data_rw <- assessment(cv_folds$splits[[i]])
  
  train_x_rw <- as.matrix(train_data_rw[, c("Close_Lag1", "Close_Lag2")])  
  train_y_rw <- train_data_rw$Target  # Log returns
  test_x_rw <- as.matrix(test_data_rw[, c("Close_Lag1", "Close_Lag2")])
  test_y_rw <- test_data_rw$Target
  
  dtrain_rw <- xgb.DMatrix(data = train_x_rw, label = train_y_rw)
  dtest_rw <- xgb.DMatrix(data = test_x_rw, label = test_y_rw)
  
  
  params <- list(objective = "reg:squarederror",
    eta = 0.1, max_depth = 5)
  
  xgb_model_rw <- xgb.train(params, dtrain_rw, nrounds = 100)
  
  pred_log_rw <- predict(xgb_model_rw, dtest_rw)
  pred_close_rw <- test_data_rw$Close_Lag1 * exp(pred_log_rw)  
  
  fold_rmse <- sqrt(mean((pred_close_rw - test_data_rw$Close)^2))
  cv_rmse <- c(cv_rmse, fold_rmse)
}

#Average RMSE across all folds
mean_cv_rmse <- mean(cv_rmse)
cat("Average Rolling Window CV RMSE:", mean_cv_rmse, "\n")
