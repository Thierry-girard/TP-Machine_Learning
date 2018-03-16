import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


""" Question 1 """
#Construction des Fonctions d'évaluation des échantillons 
def Apprentissage(X,Y):
    appArray = []
    for i in range(0,10):
        app = X[Y==i]
        moy = np.mean(app,axis=0)
        Cov = np.cov(app.T)
        logDet = np.linalg.slogdet(Cov)
        logDet = logDet[0] * logDet[1]
        CovInv = np.linalg.inv(Cov)
        # Cas equiprobable, pas de P
        appArray.append(lambda x, logDet = logDet, CovInv = CovInv: - logDet - (x - moy).T @ CovInv @ (x - moy))
    return appArray

#Evaluation de l'échantillon par rapport aux fonctions dévaluation données
def BayesEvaluation(X, appArray):
    bestMatch = 0
    bestScore = appArray[0](X)
    #print(bestScore)
    for i in range(1,10):
        score = appArray[i](X)
        #print(score)
        if(score > bestScore):
            bestMatch = i
            bestScore = score
    return bestMatch



#Question 1-2
def Question1_2(Xapp, Yapp, Xdev, Ydev):
    #Question 1
    print("Taille de l'échantillon : ", Xdev.shape[0])
    print("Classifieur Bayésien gaussien classique")
    start_time = time()
    appArray = Apprentissage(Xapp,Yapp)
    print("START")
    erreur = 0

    for i in range(0,Xdev.shape[0]):
        result = BayesEvaluation(Xdev[i], appArray)
        #print("i =",i,""  Result = ", result," Expected = ",Y[i])
        if(result != Ydev[i]):
            erreur+=1
    tiBase = (time() - start_time)
    teBase = (erreur/Xdev.shape[0])
    print("--- " ,tiBase," seconds ---")
    print("END")
    print("Taux d\'erreur : ", teBase,"\n")



    # Question 2
    teList = []
    timeList = []
    sizeList = [i for i in range(0,50,10)]+[i for i in range(50,Xapp.shape[1],50)]

    for size in sizeList:
        print("Classifieur Bayésien ACP de taille ",size)
        start_time = time()
        pca = PCA(size)
        pca.fit(Xapp)
        XappPCA = pca.transform(Xapp)
        X = pca.transform(Xdev)
        Y = Ydev
        appArrayPCA = Apprentissage(XappPCA,Yapp)

        print("START")
        erreur = 0

        for i in range(0,X.shape[0]):
            result = BayesEvaluation(X[i], appArrayPCA)
            #print('i =',i,'  Result = ', result,'Expected = ',Y[i])
            if(result != Y[i]):
                erreur+=1
        ti = (time() - start_time)
        print("--- " , ti," seconds ---")
        print("END")
        te = (erreur/X.shape[0])
        print("Taux d'erreur : ", te, "\n")
        teList.append(te)
        timeList.append(ti)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    lgd1_1 = ax1.plot(sizeList,teList,'r-o',label="Taux d'erreur en fonction de la dimension de L'ACP")
    lgd1_2 = ax1.plot(Xdev.shape[1],teBase,'g-o',label="Taux d'erreur sans ACP")
    lgd2_1 = ax2.plot(sizeList,timeList,'b-x',label="Temps de traitemet en fonction de la dimension de L'ACP")
    lgd2_2 = ax2.plot(Xdev.shape[1],tiBase,'g-x',label="temps de traitement sans ACP")

    lgds = lgd1_1 + lgd1_2 + lgd2_1 + lgd2_2

    lbls = [lgd.get_label() for lgd in lgds]
    ax1.legend(lgds, lbls, loc=4)

    ax1.set_xlabel("Dimension de l'ACP")
    ax1.set_ylabel("Taux d'erreur", color='r')
    ax2.set_ylabel("Temps de traitement de l'échantillon \nEn seconde", color='b')

    plt.show()
    fig.savefig("question.png")


#####   Question 3 A   #####
def Question3_A(Xapp, Yapp, Xdev, Ydev):
    start_time = time()
    #Scaling des données pour le SVM
    X = preprocessing.scale(Xdev)
    Y = Ydev
    print("Taille de l'échantillon : ", X.shape[0])
    print("Classifieur Support Vector Machines")

    clf = SVC(C=100000000,cache_size=4000)
    #Scaling des données pour le SVM
    clf.fit(preprocessing.scale(Xapp), Yapp) 

    print("START")
    erreur = 0
    result = clf.predict(X)
    tiBase = (time() - start_time)

    for i in range(0,X.shape[0]):
        if(result[i] != Y[i]):
            erreur+=1

    teBase = (erreur/X.shape[0])
    print("--- " ,tiBase," seconds ---")
    print("END")
    print("Taux d\'erreur : ", teBase,"\n")


    teList = []
    timeList = []
    sizeList = [i for i in range(0,50,10)]+[i for i in range(50,Xapp.shape[1],50)]

    for size in sizeList:
        start_time = time()
        #Scaling des données pour le SVM
        pca = PCA(50)
        pca.fit(Xapp)
        XappPCA = pca.transform(Xapp)
        #Scaling des données pour le SVM
        X = preprocessing.scale(pca.transform(Xdev))
        Y = Ydev
        print("Taille de l'échantillon : ", X.shape[0])
        print("Classifieur Support Vector Machines, ACP de taille ",size)

        clf = SVC(C=100000000,cache_size=4000)
        clf.fit(preprocessing.scale(XappPCA), Yapp) 

        print("START")
        erreur = 0
        result = clf.predict(X)
        ti = (time() - start_time)

        for i in range(0,X.shape[0]):
            if(result[i] != Y[i]):
                erreur+=1

        print("--- " , ti," seconds ---")
        print("END")
        te = (erreur/X.shape[0])
        print("Taux d'erreur : ", te, "\n")
        teList.append(te)
        timeList.append(ti)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    lgd1_1 = ax1.plot(sizeList,teList,'r-o',label="Taux d'erreur en fonction de la dimension de L'ACP")
    lgd1_2 = ax1.plot(Xdev.shape[1],teBase,'g-o',label="Taux d'erreur sans ACP")
    lgd2_1 = ax2.plot(sizeList,timeList,'b-x',label="Temps de traitemet en fonction de la dimension de L'ACP")
    lgd2_2 = ax2.plot(Xdev.shape[1],tiBase,'g-x',label="temps de traitement sans ACP")

    lgds = lgd1_1 + lgd1_2 + lgd2_1 + lgd2_2

    lbls = [lgd.get_label() for lgd in lgds]
    ax1.legend(lgds, lbls, loc=2)

    ax1.set_xlabel("Dimension de l'ACP")
    ax1.set_ylabel("Taux d'erreur", color='r')
    ax2.set_ylabel("Temps de traitement de l'échantillon \nEn seconde", color='b')

    plt.show()
    fig.savefig("question3_A.png")



#####   Question 3 B   #####
def Question3_B(Xapp, Yapp, Xdev, Ydev):
    start_time = time()
    print("Taille de l'échantillon : ", Xdev.shape[0])
    print("Classifieur plus proches voisins")

    neigh = KNeighborsClassifier(n_jobs=-1)
    
    neigh.fit(Xapp,Yapp) 

    print("START")
    erreur = 0
    result = neigh.predict(Xdev)
    tiBase = (time() - start_time)

    for i in range(0,Xdev.shape[0]):
        if(result[i] != Ydev[i]):
            erreur+=1

    teBase = (erreur/Xdev.shape[0])
    print("--- " ,tiBase," seconds ---")
    print("END")
    print("Taux d\'erreur : ", teBase,"\n")


    teList = []
    timeList = []
    sizeList = [i for i in range(0,50,10)]+[i for i in range(50,Xapp.shape[1],50)]

    for size in sizeList:
        start_time = time()

        pca = PCA(50)
        pca.fit(Xapp)        
        XappPCA = pca.transform(Xapp)

        X = pca.transform(Xdev)
        Y = Ydev
        print("Taille de l'échantillon : ", X.shape[0])
        print("Classifieur plus proches voisins, ACP de taille ",size)

        neigh = KNeighborsClassifier(n_jobs=-1)
        neigh.fit(X,Y) 

        print("START")
        erreur = 0
        result = neigh.predict(X)
        ti = (time() - start_time)

        for i in range(0,X.shape[0]):
            if(result[i] != Y[i]):
                erreur+=1

        print("--- " , ti," seconds ---")
        print("END")
        te = (erreur/X.shape[0])
        print("Taux d'erreur : ", te, "\n")
        teList.append(te)
        timeList.append(ti)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    lgd1_1 = ax1.plot(sizeList,teList,'r-o',label="Taux d'erreur en fonction de la dimension de L'ACP")
    lgd1_2 = ax1.plot(Xdev.shape[1],teBase,'g-o',label="Taux d'erreur sans ACP")
    lgd2_1 = ax2.plot(sizeList,timeList,'b-x',label="Temps de traitemet en fonction de la dimension de L'ACP")
    lgd2_2 = ax2.plot(Xdev.shape[1],tiBase,'g-x',label="temps de traitement sans ACP")

    lgds = lgd1_1 + lgd1_2 + lgd2_1 + lgd2_2

    lbls = [lgd.get_label() for lgd in lgds]
    ax1.legend(lgds, lbls, loc=2)

    ax1.set_xlabel("Dimension de l'ACP")
    ax1.set_ylabel("Taux d'erreur", color='r')
    ax2.set_ylabel("Temps de traitement de l'échantillon \nEn seconde", color='b')

    plt.show()
    fig.savefig("question3_B.png")


""" Main :P """

Xapp = np.load('../data/trn_img.npy')
Yapp = np.load('../data/trn_lbl.npy')

Xdev = np.load('../data/dev_img.npy')
Ydev = np.load('../data/dev_lbl.npy')

Xtest = np.load('../data/tst_img.npy')

#Question1_2(Xapp, Yapp, Xdev, Ydev)
#Question3_A(Xapp, Yapp, Xdev, Ydev)
Question3_B(Xapp, Yapp, Xdev, Ydev)

