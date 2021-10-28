import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import re
from datetime import date
import nsepy
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")
dateparse = lambda dates: pd.datetime.strptime(dates,'%m-%Y')
####################################          Live data extraction              ###################################################
##Extracting data from nsepy package
da=date.today()
gold= pd.DataFrame(nsepy.get_history(symbol="SGBAUG24",series="GB", start=date(2016,9,1), end=da))
bond= pd.DataFrame(nsepy.get_history(symbol="IRFC",series="N2", start=date(2012,1,1), end=da))

##For gold
gold=pd.DataFrame(gold["Close"])
gold["date"]=gold.index
gold["date"]=gold['date'].astype(str)
gold[["year", "month", "day"]] = gold["date"].str.split(pat="-", expand=True)
gold['Dates'] = gold['month'].str.cat(gold['year'], sep ="-")
gold.Dates=pd.to_datetime(gold.Dates)
gold.set_index('Dates',inplace=True)
col_sgb=pd.DataFrame(gold.groupby(gold.index).Close.mean())

##For bond
bond=pd.DataFrame(bond["Close"])
bond["date"]=bond.index
bond["date"]=bond['date'].astype(str)
bond[["year", "month", "day"]] = bond["date"].str.split(pat="-", expand=True)
bond['Dates'] = bond['month'].str.cat(bond['year'], sep ="-")
bond.Dates=pd.to_datetime(bond.Dates)
bond.set_index('Dates',inplace=True)
col_bond=pd.DataFrame(bond.groupby(bond.index).Close.mean())

col_sgb.columns = ["Avg_price"]
col_bond.columns = ["Avg_price"]
#############################                 Live data  extraction end                  ###############################################

############################                  SQL connection with monthly data           ################################################ 
# Connect to the database
engine_sgb = create_engine('sqlite:///gold_database.db', echo=False)
col_sgb.to_sql('SGB', con=engine_sgb,if_exists='replace')
df_sgb = pd.read_sql('select * from SGB',engine_sgb )

df_sgb.Dates=pd.to_datetime(df_sgb.Dates)
df_sgb.set_index('Dates',inplace=True)

engine_irfcb = create_engine('sqlite:///irfcb_database.db', echo=False)
col_bond.to_sql('IRFCB', con=engine_irfcb,if_exists='replace')
df_bond = pd.read_sql('select * from IRFCB',engine_irfcb)

df_bond.Dates=pd.to_datetime(df_bond.Dates)
df_bond.set_index('Dates',inplace=True)

###############################                SQL database is created                  ################################################3             
# Differencing
diff_sgb = df_sgb - df_sgb.shift()
diff_sgb.dropna(inplace = True)
diff_bond = df_bond - df_bond.shift()
diff_bond.dropna(inplace = True)
''' Finding p, d, q hyperparameter from auto_arima '''
def arima_mod(df_, diff_,x):
    model_autoARIMA = auto_arima(diff_, start_p=0, start_q=0,
                      test='adf',       
                      max_p=3, max_q=3,
                      m=12,              
                      d=None,           
                      seasonal=True,   
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    summary_string = str(model_autoARIMA.summary())
    param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)',summary_string)
    p,d,q = int(param[0][0]) , int(param[0][1]) , int(param[0][2])

    ''' ARIMA model '''
    model = ARIMA(df_, order=(p,1,q), trend = "t")
    fitted = model.fit()
    ''' Forecating '''
    forecasted_val = fitted.predict(1,len(df_)+(x-1)) 
    forecast_next= fitted.forecast(steps = x, alpha = 0.05) 

    forecast = fitted.get_forecast(x, alpha = 0.05)
    conf = forecast.conf_int(alpha=0.05)
    lower_series =pd.Series(conf["lower Avg_price"], index=forecasted_val.index[(len(df_) - 1):])
    upper_series =pd.Series(conf["upper Avg_price"], index=forecasted_val.index[(len(df_) - 1):])
    
    Forecast_series_ = pd.concat([lower_series,forecast_next,upper_series], axis = 1)
    Forecast_series_.columns = ["Lower Price","Forecasted Price","Upper Price"]
    
    return Forecast_series_
''' Calculating Returns '''
last_sgb = df_sgb['Avg_price'].iloc[-1]
last_bond = df_bond['Avg_price'].iloc[-1]
def cal_ret(x,store_s,store_b,last_s,last_b):
    if x == 4:
        return_sgb = (store_s.iloc[3] - last_s )
        return_bond = ( store_b.iloc[3] - last_b)
    elif x == 6:
        return_sgb = [(store_s.iloc[5]- last_s) + 42.886]
        return_bond = (store_b.iloc[5]- last_b)
    else:
        return_sgb = [(store_s.iloc[11]- last_s) + 85.7725] 
        return_bond = [(store_b.iloc[11]- last_b) + 81.0]
    return return_sgb, return_bond

def output_(x,y,t):
    if x > y:
       a = "The returns of SGB is "+str(x)+" and the returns of IRFC Bond is "+str(y)+" after "+str(t)+" months"
    else:
       a = "The returns of IRFC Bond is "+str(y) +" and the returns of SGB is "+str(x)+" after "+str(t)+" months"
    return a
