library(quantmod);
library(tseries);
library(timeSeries);
library(forecast);
library(xts);

#################################################################################################
# Pull data from Yahoo finance 
#enter the start date and end date
#RELIANCE.NS , SBIN.NS, BHEL.NS, TATAMOTORS.NS, INFY.NS
ticker = 'RELIANCE.NS'
train_data = getSymbols(ticker, from='2012-01-01', to='2018-05-31',auto.assign = FALSE) 
#omit na from dataframe
train_data = na.omit(train_data)
#get testing start date and end date
test_data = getSymbols(ticker, from='2018-06-01', to='2018-06-01',auto.assign = FALSE)
#omit na from test dataset
test_data = na.omit(test_data)

#################################################################################################
# Select the relevant close price series
stock_prices = train_data[,4]
actual_prices = test_data[,4]
#plot stock close price for train data
plot(stock_prices,type='l', main='Close Prices plot')

#################################################################################################
#Auto Fit Arima Model
ARIMAfit = auto.arima(log10(stock_prices), approximation=T, trace=FALSE, allowdrift=T)
#summary of ARIMA Model
summary(ARIMAfit)

#################################################################################################
#Prediction and comparison with actual prices
par(mfrow = c(1,1))
#forecasting future values from model # specify no. of days ahead to predict
pred = forecast(ARIMAfit, h =1)
#converting back to original format
predicted_prices = as.data.frame(10^(as.vector(pred$mean)))
#getting actual price dataframe
actual_prices = as.data.frame(actual_prices)
#specifying rownames of actual price to merge the actual and predicted values
rownames(actual_prices) <- 1:nrow(actual_prices)
#merging both the dataframes
merged_prices = merge(actual_prices,predicted_prices,by = 0,all.y=TRUE)[-1]
#changing column names of predicted and actual price values
colnames(merged_prices) <- c("actual prices","predicted prices")

#################################################################################################
#Plot ACF and PACF of residuals
par(mfrow=c(1,2))
acf(ts(ARIMAfit$residuals),na.action = na.pass,main='ACF Residual')
pacf(ts(ARIMAfit$residuals),na.action = na.pass,main='PACF Residual')
#print the adf results for residuals
print(adf.test(ts(ARIMAfit$residuals)))

#################################################################################################
#Finding RMSE
merged_prices$sq_error = (merged_prices[,1]-merged_prices[,2])^2
RMSE  <- (sum(merged_prices$sq_error)/nrow(merged_prices))^0.5
print("RMSE : ")
print(RMSE)

#################################################################################################
"
#######Manual Selection of ARIMA MODELS

#Finding Differencing Component
plot(diff(stock_prices),ylab='Differenced Closing Prices')

#Log-Transform Data Plot
plot(log10(stock_prices),ylab='Log (Closing Prices)')

#Plot Log differenced data
plot(diff(log10(stock_prices)),ylab='Differenced Log (Closing price)')

#Constructing ACF and PACF plots
par(mfrow = c(1,2))
acf(ts(diff(log10(stock_prices))),na.action = na.pass,main='ACF Closing Prices')
pacf(ts(diff(log10(stock_prices))),na.action = na.pass,main='PACF Closing Prices')
"

