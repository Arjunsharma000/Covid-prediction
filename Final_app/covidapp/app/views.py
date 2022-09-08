from django.shortcuts import render
import datetime as dt
import pickle as pkl 
import pandas as pd 
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'index.html')
def load(filename):
    file=open(filename, 'rb')
    data=pkl.load(file)
    file.close()
    return data


def predict(request):
    model=load('linear (1).pkl')
    lab=load("labelencoded (2).pkl")
    date = request.GET['date']
    # date = dt.datetime.strptime(date, "%Y-%m-%d")
    date = np.array([date])
    date = pd.to_datetime(date)
    date = lab.transform(date)
    print(date)
    date = np.array([date]).reshape(1,-1)
    
    # print(date)
    pred=model.predict(date)
    return render(request,"predict.html",{"prediction":abs(int(pred[0][0]))})

   