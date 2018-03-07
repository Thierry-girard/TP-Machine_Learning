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
    for i in range(0,10):
        app = X[Y==i]
        moy = np.mean(app,axis=0)
        Cov = np.cov(app.T)
        logDet = np.linalg.slogdet(Cov)
        logDet = logDet[0] * logDet[1]
        CovInv = np.linalg.inv(Cov)
        appArray.append(lambda x, logDet = logDet, CovInv = CovInv: - logDet - (x - moy).T @ CovInv @ (x - moy))
    return appArray

def BayesEvaluation(x,appArray):   
    bestMatch = 0
    bestScore = appArray[0](x)
    #print(bestScore)
    for i in range(1,10):
        score = appArray[i](x)
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

for i in range(0,X0.shape[0]):
    result = BayesEvaluation(X0[i], appArray)
    print('i =',i,'  Result = ', result,'Expected = ',Y0[i])
    if(result != Y0[i]):
        tauxErreur+=1

print("--- %s seconds ---" % (time() - start_time))
print('Taux d\'erreur : ', tauxErreur/X0.shape[0] )



"""img = X[imgIndex].reshape(32,24)
plt.imshow(img, plt.cm.gray)
plt.show()"""