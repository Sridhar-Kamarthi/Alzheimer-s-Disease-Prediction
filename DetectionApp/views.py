from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

global username, dataset
global X_train, y_train, X_test, y_test, labels, X, Y, vt, scaler, dataset, cognitive_features
global accuracy, precision, recall, fscore

#function to calculate all metrics
def calculateMetrics(y_test, predict):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    

def getCognitiveFeatures(X, n_components=25, n_neighbors=10):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    correlation_probabilities = np.zeros_like(distances)
    for i in range(X.shape[0]):
        for j in range(n_neighbors):
            correlation_probabilities[i, j] = 1 / n_neighbors
   
    def nca_objective(W):
        loss = 0
        for i in range(X.shape[0]):
            for j in range(n_neighbors):
                neighbor_index = indices[i, j]
                loss += correlation_probabilities[i, j] * np.dot(X[i] - X[neighbor_index], W.T) @ (X[i] - X[neighbor_index])
        return loss 
    
    # Use PCA to solve the NCA optimization problem
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(X)
    return pca, transformed_data


def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global scaler, vt, cognitive_features
        labels = ['No Alzheimer', 'Alzheimer Detected']
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("DetectionApp/static/"+fname):
            os.remove("DetectionApp/static/"+fname)
        with open("DetectionApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        testdata = pd.read_csv("DetectionApp/static/"+fname)
        temp = testdata.values
        testdata.fillna(0, inplace = True)#replace missing values
        testdata.drop(['PatientID', 'DoctorInCharge'], axis = 1,inplace=True)
        testdata = testdata.values
        testdata = scaler.transform(testdata)
        testdata = cognitive_features.transform(testdata)
        predict = vt.predict(testdata)
        predict = predict.ravel()
        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Data</th>'
        output += '<th><font size="3" color="black">Land Price Prediction</th></tr>'
        for i in range(len(predict)):
            output += '<tr><td><font size="3" color="black">'+str(temp[i])+'</td>'
            if predict[i] == 0:
                output += '<td><font size="3" color="green">'+str(labels[predict[i]])+'</font></td></tr>'
            else:
                output += '<td><font size="3" color="red">'+str(labels[predict[i]])+'</font></td></tr>'
        output+= "</table></br></br></br></br>"    
        context= {'data':output}
        return render(request, 'UserScreen.html', context)        

def FeaturesExtraction(request):
    if request.method == 'GET':
        global X_train, y_train, X_test, y_test, labels, X, Y, scaler, dataset, cognitive_features
        dataset.fillna(0, inplace = True)#replace missing values
        Y = dataset['Diagnosis'].ravel()
        dataset.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis = 1,inplace=True)
        X = dataset.values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)#shuffle dataset values
        X = X[indices]
        Y = Y[indices]
        output= "<font size=3 color=blue>Dataset processing completed</font><br/>"
        output+= "<font size=3 color=blue>Total records found in dataset = "+str(X.shape[0])+"</font><br/>"
        output+= "<font size=3 color=blue>Total features found in dataset before applying Cognitive Features Selection = "+str(X.shape[1])+"</font><br/><br/>"
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        output1 = "<font size=3 color=blue>Train & Test Split Details</font><br/>"
        output1+= "<font size=3 color=blue>80% dataset used to train algorithms = "+str(X_train.shape[0])+"</font><br/>"
        output1+= "<font size=3 color=blue>20% dataset used to test algorithms = "+str(X_test.shape[0])+"</font><br/>"
        data = np.load("model/data.npy", allow_pickle=True)
        X_train, X_test, y_train, y_test = data
        cognitive_features, X_train = getCognitiveFeatures(X_train)
        X_test = cognitive_features.transform(X_test)
        output+= "<font size=3 color=blue>Total features available in dataset after applying Cognitive Features Selection = "+str(X_train.shape[1])+"</font><br/><br/>"
        output += output1
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def LoadDatasetAction(request):    
    if request.method == 'POST':
        global dataset
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("DetectionApp/static/"+fname):
            os.remove("DetectionApp/static/"+fname)
        with open("DetectionApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("DetectionApp/static/"+fname)
        columns = dataset.columns
        data = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(0,100):
            output += '<tr>'
            for j in range(len(data[i])):
                output += '<td><font size="3" color="black">'+str(data[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RunML(request):
    if request.method == 'GET':
        global vt, X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore
        accuracy = []
        precision = []
        recall = []
        fscore = []

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        calculateMetrics(y_test, predict)

        knn = KNeighborsClassifier(n_neighbors=3)
        dt = DecisionTreeClassifier()
        rf = RandomForestClassifier() 
        nb = GaussianNB()
        lr = LogisticRegression()
        ada = AdaBoostClassifier()
        xgb_cls = XGBClassifier()
        estimators = [('knn', knn), ('dt', dt), ('rf', rf), ('nb', nb), ('lr', lr), ('ada', ada), ('xg', xgb_cls)]
        vt = VotingClassifier(estimators=estimators)
        vt.fit(X_train, y_train)
        predict = vt.predict(X_test)
        calculateMetrics(y_test, predict)

        output='<table border=1 align=center width=100%><tr>'
        columns = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE']
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        columns = ['Existing Non-Ensemble Algorithm', 'Proposed Voting Based Ensemble Algorithm']
        for i in range(len(accuracy)):
            output += '<tr><td><font size="3" color="black">'+columns[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output += '</table><br/>'
        df = pd.DataFrame([['Non-Ensemble','Accuracy',accuracy[0]],['Non-Ensemble','Precision',precision[0]],['Non-Ensemble','Recall',recall[0]],['Non-Ensemble','FSCORE',fscore[0]],
                           ['Voting Based Ensemble','Accuracy',accuracy[1]],['Voting Based Ensemble','Precision',precision[1]],['Voting Based Ensemble','Recall',recall[1]],['Voting Based Ensemble','FSCORE',fscore[1]],
                          ],columns=['Parameters','Algorithms','Value'])
        df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
        return render(request, 'LoadDataset.html', {})

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'alzheimer',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
               
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'alzheimer',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'alzheimer',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Alzheimer Disease prediction"
        context= {'data':output}
        return render(request, 'Register.html', context)        

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})         

def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

