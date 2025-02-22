library(quantmod)
library(Hmisc)
library(tidyverse)
library(fpp2)
library(tseries)

ticker <- "AMGN"
df_stock <- getSymbols.yahoo(ticker, periodicity = "weekly", from = "2014-01-01", auto.assign = FALSE)[,4]
head(df_stock, n = 10)
tail(df_stock, n = 10)
describe(df_stock) #observation: 580
str(df_stock)

#Checking for NA values
(sum(is.na(df_stock))) #No NA values

#Plotting time series
autoplot(df_stock) + 
  ylab("AMGN Closing Price") + xlab("Year") #initial assumption - trend but no seasonality

#stabilizing variance
log(df_stock) %>% autoplot()
df_stock %>% BoxCox(lambda = BoxCox.lambda(df_stock)) %>% autoplot()
#No effect so variance stabilization is not necessary

###Making train and test data sets
set.seed(843)
split_date <- index(df_stock[floor(0.8 * length(df_stock))])  #the date at 80% split
train <- df_stock[index(df_stock) < split_date]
test <- df_stock[index(df_stock) >= split_date]


##Checking for stationarity 
#stabilizing mean
adf_test <- adf.test(df_stock) #p value is very small suggesting that differencing is not required
df_stock %>% ndiffs() # shows that differencing of 1 is required

train_diff <- train %>% diff()
train_diff %>% autoplot()


ggAcf(train) + ggtitle("ACF plot of AMGN")
ggPacf(train) + ggtitle("PACF plot of AMGN")# q value is zero

#Model fitting
model_opt <- Arima(train, order = c(1,0,0), include.constant = TRUE)
model_auto <- auto.arima(train, stepwise = FALSE, approximation = FALSE, trace = TRUE, ic = "bic")
model_naive <- naive(train)

aic_values <- data.frame()
for(d in 0:1) {
  for (p in 0:5){
    for (q in 0:2){
      test_fit <- Arima(train, order = c(p,d,q), method = "ML")
      AIC <- AIC(test_fit)
      aic_values <- rbind(aic_values, data.frame(p,d,q,AIC))
    }
  }
}

print(min(aic_values$AIC))
aic_values %>% slice_min(AIC)

model_aic <- Arima(train, order = c(3,1,2))

summary(model_opt)
summary(model_auto)
summary(model_naive)
summary(model_aic)

#Checking residual
checkresiduals(model_opt)
checkresiduals(model_auto)
checkresiduals(model_naive)
checkresiduals(model_aic)

Box.test(model_opt$residuals, lag = 10, type = "Ljung-Box")
Box.test(model_opt$residuals, lag = 20, type = "Ljung-Box")
Box.test(model_auto$residuals, lag = 10, type = "Ljung-Box")
Box.test(model_auto$residuals, lag = 20, type = "Ljung-Box")
Box.test(model_aic$residuals, lag = 10, type = "Ljung-Box")
Box.test(model_aic$residuals, lag = 20, type = "Ljung-Box")

#forecast evaluation
opt_forecast <- forecast(model_opt, h = length(test))
auto_forecast <- forecast(model_auto, h = length(test))
naive_forecast <- naive(train, h = length(test))
aic_forecast <- forecast(model_aic, h = length(test))

print(acc_opt <- accuracy(opt_forecast, test))
print(acc_auto <- accuracy(auto_forecast, test))
print(acc_naive <- accuracy(naive_forecast, test))
print(acc_aic <- accuracy(aic_forecast, test))


#forecast visualization - auto.arima has the lowest RMSE, not so different from naive.
#So, a random model has similar performance to arima
forecast_df <- data.frame(
  Date = index(test),  
  Forecast = as.numeric(auto_forecast$mean),  
  Lower_80 = as.numeric(auto_forecast$lower[, 1]),  
  Upper_80 = as.numeric(auto_forecast$upper[, 1]),  
  Lower_95 = as.numeric(auto_forecast$lower[, 2]),  
  Upper_95 = as.numeric(auto_forecast$upper[, 2])   
)

#Converting test set (xts) to data frame
test_df <- data.frame(
  Date = index(test),
  Actual = as.numeric(test)
)

#Merging forecast and actual data
plot_data <- full_join(forecast_df, test_df, by = "Date") %>%
  pivot_longer(cols = c("Forecast", "Actual"), names_to = "Series", values_to = "Value")

#Plotting with ggplot
ggplot(plot_data, aes(x = Date, y = Value, color = Series)) +
  geom_line(size = 1) +  
  geom_ribbon(data = forecast_df, aes(x = Date, ymin = Lower_80, ymax = Upper_80), 
              inherit.aes = FALSE, fill = "blue", alpha = 0.2) +  
  geom_ribbon(data = forecast_df, aes(x = Date, ymin = Lower_95, ymax = Upper_95), 
              inherit.aes = FALSE, fill = "blue", alpha = 0.1) +  
  xlab("Year") +
  ylab("Stock Price") +
  ggtitle("AMGN Stock Price Forecast vs Actual") +
  theme_minimal()

