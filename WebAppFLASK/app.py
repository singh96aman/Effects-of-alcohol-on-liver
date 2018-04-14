from flask import Flask, session, redirect, url_for, escape, request, render_template, jsonify

import numpy
import pylab as ply

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score

from pandas import DataFrame, read_csv, concat
import matplotlib.pyplot as plot
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/liver', methods=['GET','POST'])
def Liver():
    if request.method == 'POST':
        mcv=request.form['mcv']
        alkphos=request.form['alkphos']
        sgpt=request.form['sgpt']
        sgot=request.form['sgot']
        gammagt=request.form['gammagt']
        drinks=request.form['drinks']
        temp,result,coeff,test=TEST(mcv, alkphos, sgpt, sgot, gammagt, drinks)
        return render_template('output.html',temp=temp,res=result,coeff=coeff,test=test)
    return render_template('input.html')

def data_preprocessing():
    print "Prepocessing data : Reading data into a DataFrame"
    dataset = read_csv("./bupa.data", sep=',', header=None)
    dataset['id'] = range(1, len(dataset)+1)
    lab = LabelEncoder()
    lab.fit(dataset[6].drop_duplicates())
    dataset[6] = lab.transform(dataset[6])
    class_names = list(lab.classes_)
    return (dataset, class_names)

def TEST(mcv, alkphos, sgpt, sgot, gammagt, drinks):
    dataset, class_names = data_preprocessing()

    target = dataset[6]
    train = dataset.drop(['id', 6], axis = 1)

    model_rfc = RandomForestClassifier(n_estimators = 625, criterion = "entropy", n_jobs = -1)

    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size=0.25)
    probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)


    test1 = int(mcv)
    test2 = int(alkphos)
    test3 = int(sgpt)
    test4 = int(sgot)
    test5 = int(gammagt)
    test6 = float(drinks)

    test = test1,test2,test3,test4,test5,test6

    result = model_rfc.predict([[test1,test2,test3,test4,test5,test6]])

    if result == 0:
        temp = "You are not suffering from any Liver disease or disorder."
    else :
        temp = "You are suffering from Liver disease or disorder because of your excessive drinking. PLEASE CONSULT A HEPATOLOGIST !"

    roc_acc=float(roc_acc)*100

    return temp,roc_acc,model_rfc.feature_importances_,test


app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == '__main__':
    app.run(port=5502)

