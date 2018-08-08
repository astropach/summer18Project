library(quantmod);
library(tseries);
library(timeSeries);
library(forecast);
library(xts);

#################################################################################################
#Pull data from Yahoo finance 
#enter the start date and end date
train_data = getSymbols('RELIANCE.NS', from='2017-05-31', to='2018-05-31',auto.assign = FALSE) 
#omit na from dataframe
train_data = na.omit(train_data)
#get testing start date and end date
test_data = getSymbols('RELIANCE.NS', from='2018-06-01', to='2018-06-30',auto.assign = FALSE)
#omit na from test dataset
test_data = na.omit(test_data)

#################################################################################################
# Select the relevant close price series
stock_prices = train_data[,4]
colnames(stock_prices) <- c("ClosePrice")
actual_prices = test_data[,4]
colnames(actual_prices) <- c("ClosePrice")
#plot stock close price for train data
plot(stock_prices,type='l', main='Close Prices plot')

#################################################################################################
#Calculating Percnt daily returns
stock_prices$PDR <- diff(log(stock_prices$ClosePrice))

#################################################################################################
#Calculating average PDR, variance of PDR and Std Dev of PDR and Drift for GBM
PDR.mean <- as.numeric(mean(stock_prices$PDR , na.rm = TRUE))
PDR.var <- as.numeric(var(stock_prices$PDR , na.rm = TRUE))
PDR.sd <- as.numeric(sd(stock_prices$PDR , na.rm = TRUE))
drift <- as.numeric(PDR.mean - (PDR.var/2))

#################################################################################################
#Simulation of GBM for stock prediction
last_day_price = as.numeric(stock_prices$ClosePrice[length(stock_prices$ClosePrice)])

n_sim = 20
sim_matrix <- matrix(0, nrow = n_sim, ncol = length(actual_prices$ClosePrice))
colnames(sim_matrix) <- c(1:length(actual_prices$ClosePrice))

for (i in 1:n_sim){
  for (j in 1:length(actual_prices$ClosePrice)){
    if (j==1){
      sim_matrix[i,j] = last_day_price*exp(drift + PDR.sd*(rnorm(1)))
    }else{
      sim_matrix[i,j] = as.numeric(actual_prices[j-1,1])*exp(drift + PDR.sd*(rnorm(1)))
    }
  }
}

#################################################################################################
#Calculating Average values for simulation
actual_prices$Avg_Sim_Price <- 1
for (i in 1:length(actual_prices$ClosePrice)){
  actual_prices[i,2] = as.numeric(mean(sim_matrix[,i]))
}

#################################################################################################
#Calculating RMSE metric
sq_error <- (actual_prices$ClosePrice - actual_prices$Avg_Sim_Price)^2
RMSE <- (sum(sq_error)/length(sq_error))^0.5
print ("RMSE:")
print(RMSE)

#################################################################################################
#Plotting Graphs
plot(1:length(actual_prices$ClosePrice),actual_prices$ClosePrice,type="l",col="red")
lines(1:length(actual_prices$ClosePrice),actual_prices$Avg_Sim_Price,type="l",col="green")

