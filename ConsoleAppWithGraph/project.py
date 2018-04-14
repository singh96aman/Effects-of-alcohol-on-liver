'''
*******************************************************************************************************
                        
                                Date Created : 10th April, 2018
                        Project Title : Case Study on Effects of Alcohol on Liver
                                Author : Aman Singh Thakur

*******************************************************************************************************
'''

##################### IMPORT NECESSARY PYTHON LIBRARIES #####################

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


'''
*******************************************************************************************************
                                        START OF PROGRAM
*******************************************************************************************************
'''

##################### FETCHING AND PREPROCESSING DATA #####################

def data_preprocessing():
    print "Prepocessing data : Reading data into a DataFrame"
    dataset = read_csv("./bupa.data", sep=',', header=None)
    dataset['id'] = range(1, len(dataset)+1)
    lab = LabelEncoder()
    lab.fit(dataset[6].drop_duplicates())
    dataset[6] = lab.transform(dataset[6])
    class_names = list(lab.classes_)
    return (dataset, class_names)

##################### Plot Graph Between ALL CLASSES #####################

def GRAPH_CLASS():
    dataset, class_names = data_preprocessing()
    print "Plotting Graph Between ALL CLASSES"
    title = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
    for i in range(0,6):
        for j in range(i,6):
            if i is not j :
                f = plot.figure(figsize=(8, 6))
                colors = ['b', 'y']
                for k in range(0, 2):
                    plot.scatter(dataset[dataset[6] == k][i], dataset[dataset[6] == k][j], c=colors[k], label="%s" % class_names[k])
                plot.legend()
                plot.title('%s (1) VS %s (2)' % (title[i],title[j]))
                f.savefig('./%s/%s_%s.png' % (directories[0],title[i],title[j]))
                plot.close()

##################### Plot Graph for all classes #####################

def CLASS_PLOT():
    dataset, class_names = data_preprocessing()
    title = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
    for k in range(0,6):
        print "Plotting %s class" % title[k]
        fig, axes = plot.subplots(ncols=1)
        e = dataset.pivot_table('id', [k], 6, 'count').plot(ax=axes, title='%s' % title[k])
        f = e.get_figure()
        f.savefig('./%s/%s.png' % (directories[0], title[k]))
        plot.close()

#####################  Plot BOX PLOTS for all classes #####################

def BOX_PLOT():
    dataset, class_names = data_preprocessing()
    title = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
    dataset['class'] = dataset[6]
    for k in range(0, 6):
        print "Plotting Box for %s by class" % title[k]
        df = concat([dataset[k], dataset['class']], axis=1, keys=[title[k], 'class'])
        f = plot.figure(figsize=(8, 6))
        p = df.boxplot(by='class', ax = f.gca())
        f.savefig('./%s/%s_box.png' % (directories[0], title[k]))
        plot.close()

##################### ANALYZING CLASSES AND PLOTTING ROC #####################

def ROC():
    print "\n"
    dataset, class_names = data_preprocessing()
    target = dataset[6]
    train = dataset.drop(['id', 6], axis = 1)

    print "Preparing Models for :\n1.\tLogistic Regression\n2.\tGradient Boosting Classifier\n3.\tK-Neighbors Classifier\n4.\tRandom Forest Classifier"

    model_rfc = RandomForestClassifier(n_estimators = 100, criterion = "entropy", n_jobs = -1)
    model_knc = KNeighborsClassifier(n_neighbors = 10, algorithm = 'brute')
    model_gbc = GradientBoostingClassifier(n_estimators = 100)
    model_lr = LogisticRegression(penalty='l1', tol=0.01)

    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size=0.25)

    print "\n**COMPLETING ANALYSIS**\n"

    ply.clf()
    plot.figure(figsize=(8,6))

    # LogisticRegression
    probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)
    print "Using LogisticRegression, We get accuracy : ",float(roc_acc)*100
    ply.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_acc))

    # GradientBoostingClassifier
    probas = model_gbc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)
    print "Using Gradient Boosting Classifier, We get accuracy : ",float(roc_acc)*100
    ply.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('GradientBoosting',roc_acc))

    # RandomForestClassifier
    probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)
    print "Using Random Forest Classifier, We get accuracy : ",float(roc_acc)*100
    ply.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandomForest',roc_acc))


    # KNeighborsClassifier
    probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)
    print "Using KNeighbors Classifier, We get accuracy : ",float(roc_acc)*100
    ply.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_acc))

##################### Plotting ROC graph for all classes in alogrithm_analysis folder #####################

    ply.plot([0, 1], [0, 1], 'k--')
    ply.xlim([0.0, 1.0])
    ply.ylim([0.0, 1.0])
    ply.xlabel('False Positive Rate')
    ply.ylabel('True Positive Rate')
    ply.legend(loc=0, fontsize='small')
    ply.savefig('./%s/ROC_ALL.png' % directories[1])

##################### Comparing ROC between Random forest and Gradient Boosting ##################### 

def COMPARE():
    dataset, class_names = data_preprocessing()

    print "\nPlotting ROC for Random Forest vs Gradient Descent"

    target = dataset[6]
    train = dataset.drop(['id', 6], axis = 1)

    model_rfc = RandomForestClassifier(n_estimators = 625, criterion = "entropy", n_jobs = -1)
    model_gbc = GradientBoostingClassifier(n_estimators = 625)

    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size=0.25)

    ply.clf()
    plot.figure(figsize=(8,6))

    # RandomForestClassifier
    probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)
    ply.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandomForest',roc_acc))

    # GradientBoostingClassifier
    probas = model_gbc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)
    ply.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('GradientBoosting',roc_acc))

    ply.plot([0, 1], [0, 1], 'k--')
    ply.xlim([0.0, 1.0])
    ply.ylim([0.0, 1.0])
    ply.xlabel('False Positive Rate')
    ply.ylabel('True Positive Rate')
    ply.legend(loc=0, fontsize='small')
    ply.savefig('./%s/compare.png' % directories[1])

##################### TEST FUNCTION TO MAKE IT USER FRIENDLY #####################

def TEST():
    dataset, class_names = data_preprocessing()

    print "\nInside the TEST function, Training Random Forest Classifier"

    target = dataset[6]
    train = dataset.drop(['id', 6], axis = 1)

    model_rfc = RandomForestClassifier(n_estimators = 625, criterion = "entropy", n_jobs = -1)

    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size=0.25)
    probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_acc  = auc(fpr, tpr)

    print "All the coefficients after training are ",model_rfc.feature_importances_

    print "\nPlease Input the following Test Results :"

    test1 = int(raw_input("mcv mean corpuscular volume : "))
    test2 = int(raw_input("alkphos alkaline phosphotase : "))
    test3 = int(raw_input("sgpt alamine aminotransferase: "))
    test4 = int(raw_input("sgot aspartate aminotransferase: "))
    test5 = int(raw_input("gammagt gamma-glutamyl transpeptidase : "))
    test6 = float(raw_input("number of half-pint equivalents of alcoholic beverages drunk per day : "))

    result = model_rfc.predict([[test1,test2,test3,test4,test5,test6]])

    if result == 0:
        print "\nYou are not suffering from any Liver disease or disorder. Predicted with an accuracy of ",float(roc_acc)*100
    else :
        print "\nYou are suffering from Liver disease or disorder because of your excessive drinking. Predicted with an accuracy of ",float(roc_acc)*100
        print "PLEASE CONSULT A HEPATOLOGIST !"


################## CREATE DIRECTORIES FOR GRAPH AND BOX ANALYSIS & ALGORITHM ANALYSIS ##################

directories = ['graph_analysis', 'algorithm_analysis']

for check in directories:
        if not os.path.exists(check):
            os.makedirs(check)

################## FUNCTION CALL ######################

GRAPH_CLASS()
CLASS_PLOT()
BOX_PLOT()
ROC()
COMPARE()
TEST()
print "End of program"

'''
*******************************************************************************************************
                                        END OF PROGRAM
*******************************************************************************************************
'''
