from flask import Flask, render_template, request
from model import arima_mod,diff_bond,diff_sgb,cal_ret,output_
from model import df_sgb,df_bond,last_sgb,last_bond
import pandas as pd

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('Home.html')

@app.route("/Predict")
def predict():
    return render_template('Predict.html')

@app.route("/Output", methods =['POST'])
def Output():
    # HTML -> .py
    if request.method == "POST":
        n = int(request.form["Time Period"])
        store_sgb = arima_mod(df_sgb, diff_sgb,n)
        store_bond = arima_mod(df_bond,diff_bond,n)
        gain_sgb,gain_bond = cal_ret(n,store_sgb["Forecasted Price"],store_bond["Forecasted Price"],last_sgb,last_bond)
        data = output_(gain_sgb,gain_bond,n)
        dfstore_sgb = pd.DataFrame(store_sgb)
        dfstore_bond = pd.DataFrame(store_bond)

        #plotbonds = store_bond.plot()
    return render_template('Output.html', output_sgb = [dfstore_sgb.to_html(classes = "data")], output_bond = [dfstore_bond.to_html(classes = "data")] , header="true" , text = data) 

if __name__ == "__main__":
    app.run(debug=True, port = 8000)

#connect virtualenv'''.\env\Scripts\activate.ps1 '''
#RunPythonflask'''python .\app.py'''
#Home'''http://127.0.0.1:8000/'''
