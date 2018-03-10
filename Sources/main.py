import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA

Xapp = np.load('../data/trn_img.npy')
Yapp = np.load('../data/trn_lbl.npy')

Xdev = np.load('../data/dev_img.npy')
Ydev = np.load('../data/dev_lbl.npy')

Xtest = np.load('../data/tst_img.npy')


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


""" Main :P """

#Question 1
X = Xdev
Y = Ydev
print("Taille de l'échantillon : ", X.shape[0])
print("Classifieur Bayésien gaussien classique")
appArray = Apprentissage(Xapp,Yapp)
print("START")
tauxErreur = 0
start_time = time()

for i in range(0,X.shape[0]):
    result = BayesEvaluation(X[i], appArray)
    #print("i =",i,""  Result = ", result," Expected = ",Y[i])
    if(result != Y[i]):
        tauxErreur+=1
print("--- " ,(time() - start_time)," seconds ---")
print("END")
print("Taux d\'erreur : ", tauxErreur/X.shape[0],"\n")


# Question 2
teList = []
timeList = []
sizeList = range(50, Xdev.shape[1], 50)

#for size in range(50, Xdev.shape[1], 50):
for size in sizeList:
    print("Classifieur Bayésien ACP de taille ",size)
    pca = PCA(size)
    pca.fit(Xapp)
    XappPCA = pca.transform(Xapp)
    X = pca.transform(Xdev)
    Y = Ydev
    appArrayPCA = Apprentissage(XappPCA,Yapp)

    print("START")
    erreur = 0
    start_time = time()

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
ax1.plot(sizeList,teList,'r-o' )
ax2.plot(sizeList,timeList,'b-x' )

ax1.set_xlabel("Dimension de l'ACP")
ax1.set_ylabel("Taux d'erreur", color='r')
ax2.set_ylabel("Temps de traitement de l'échantillon \n(sans l'apprentissage)\nEn seconde", color='b')

plt.show()

"""img = X[imgIndex].reshape(32,24)
plt.imshow(img, plt.cm.gray)
plt.show()

plt.scatter(*zip(*testList2))
plt.title('Random Figure')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()"""