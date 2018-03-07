import numpy as np
import matplotlib.pyplot as plt
from time import time

X0 = np.load('../data/trn_img.npy')
Y0 = np.load('../data/trn_lbl.npy')

X = np.load('../data/dev_img.npy')
Y = np.load('../data/dev_lbl.npy')


def Apprentissage(X,Y):
    # Cas equiprobable, pas de P
    appArray = []
    for i in range(0,9):
        app = X[Y==i]
        moy = np.mean(app,axis=0)
        Cov = np.cov(app.T)
        appArray.append((moy,Cov))       
    return appArray

def BayesEvaluation(X,appArray):   
    bestMatch = 0
    bestScore = - np.Inf
    for i in range(0,9):
        moy = appArray[i][0]
        Cov = appArray[i][1]
        logDet = np.linalg.slogdet(Cov)
        score = - logDet[0] * logDet[1] - (X - moy).T @ np.linalg.inv(Cov) @ (X - moy)
        #print(score)
        if(score > bestScore):
            bestMatch = i
            bestScore = score
    return bestMatch


appArray = Apprentissage(X0,Y0)

imgIndex = 160

print('taille de l\'échantillon', X.shape[0])
print('START')
tauxErreur = 0
start_time = time()

for i in range(0,X.shape[0]):
    result = BayesEvaluation(X[i], appArray)
    print('I =',i,'  Result = ',result ,'Expected = ',Y[i])
    if(result != Y[i]):
        tauxErreur+=1

print("--- %s seconds ---" % (time() - start_time))
print('Taux d\'erreur : ', tauxErreur/X.shape[0] )



"""img = X[imgIndex].reshape(32,24)
plt.imshow(img, plt.cm.gray)
plt.show()"""