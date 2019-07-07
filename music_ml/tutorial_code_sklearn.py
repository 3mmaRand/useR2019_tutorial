#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:22:06 2019

tutorial code for sklearn part of stats/machine learning session

@author: mjknight
"""

# %% read in the data
import os
import pandas as pd

dirs_to_use = ["Violin", "Piano", "Violin_and_Piano"]

df_seg = None
df_info = None
for d in dirs_to_use:
    for f in os.listdir(d):
        if f.endswith("segments.xlsx"):
            if df_seg is None:
                df_seg = pd.read_excel(os.path.join(d, f))
            else:
                df_seg = df_seg.append(pd.read_excel(os.path.join(d, f)),
                                       ignore_index=True)
        elif f.endswith("SegmentInfo.xlsx"):
            if df_info is None:
                df_info = pd.read_excel(os.path.join(d, f))
                df_info["Instrument"] = pd.Series([d]*len(df_info),
                                                  index=df_info.index)
            else:
                df = pd.read_excel(os.path.join(d, f))
                df["Instrument"] = pd.Series([d]*len(df), index=df.index)
                df_info = df_info.append(df, ignore_index=True)

# %% Apply PCA, plot some components, see variance explained etc

from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mdl= PCA()
new_data = mdl.fit_transform(df_seg)

p = df_info["Instrument"]=="Piano"
v = df_info["Instrument"]=="Violin"

plt.figure()
plt.scatter(new_data[p,0],new_data[p,1],label="Piano")
plt.scatter(new_data[v,0],new_data[v,1],label="Violin")
plt.legend()
plt.grid(b=True)
plt.savefig("pca_2d.png",dpi=300,transparent=True)

plt.figure()
plt.plot(mdl.explained_variance_)
plt.grid(b=True)
plt.savefig("explained_variance.png",dpi=300,transparent=True)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(new_data[p, 0], new_data[p, 1], new_data[p, 2],label="Piano")
ax.scatter(new_data[v, 0], new_data[v, 1], new_data[v, 2],label="Violin")
#plt.savefig("pca_3d.png",dpi=300,transparent=True)

# %% Train an SVM on raw data, plot an ROC curve

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

mdl = SVC(kernel="poly", degree=2, probability=True)

mdl.fit(df_seg,df_info["Instrument"])

proba = mdl.predict_proba(df_seg)
fpr, tpr, thresholds = roc_curve(df_info["Instrument"], proba[:, 0], pos_label="Piano")
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(b=True)
plt.savefig("ROC_2class.png",dpi=300,transparent=True)


# %% Train an SVM on raw data to classify. with split of test and training

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# limit to Piano or Violin classes for 2-class example
ix = (df_info["Instrument"]=="Piano") | (df_info["Instrument"]=="Violin")

# Split the data, 25% for testing, randomise order
X_train, X_test, y_train, y_test = train_test_split(df_seg[ix],
                                                    df_info["Instrument"][ix],
                                                    test_size=0.25,
                                                    shuffle=True)

mdl = SVC(kernel="poly", degree=2, probability=True)
mdl.fit(X_train,y_train)
pred_class = mdl.predict(X_train)
pred_class_test = mdl.predict(X_test)

cmat = confusion_matrix(y_train, pred_class, labels=["Piano", "Violin"])
rep = classification_report(y_train, pred_class, labels=["Piano", "Violin"])

cmat_test = confusion_matrix(y_test, pred_class_test, labels=["Piano", "Violin"])
rep_test = classification_report(y_test, pred_class_test, labels=["Piano", "Violin"])

prob = mdl.predict_proba(df_seg)

proba = mdl.predict_proba(X_test)

plt.figure()
for i,labels in enumerate(["Piano", "Violin"]):
    fpr, tpr, thresholds = roc_curve(y_test, proba[:, i], pos_label=labels)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=labels, linewidth=2)
    print("AUC for "+labels+" = "+str(roc_auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(b=True)
plt.legend()
plt.savefig("ROC_2class_splitData.png",dpi=300,transparent=True)




# %% Use a pipeline with StandardScaler, PCA and SVM. Cross-val

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

X_train, X_test, y_train, y_test = train_test_split(df_seg,
                                                    df_info["Instrument"],
                                                    test_size=0.5,
                                                    shuffle=True)

mdl = make_pipeline(StandardScaler(),
                    PCA(n_components=50),
                    OneVsRestClassifier(SVC(kernel="poly",
                                            degree=3,
                                            probability=True)))
mdl.fit(X_train,y_train)

pred_class_test = mdl.predict(X_test)

proba = mdl.predict_proba(X_test)

cmat = confusion_matrix(y_test, pred_class_test, labels=mdl.classes_)
rep_test = classification_report(y_test, pred_class_test, labels=mdl.classes_)

plt.figure()
for i,labels in enumerate(mdl.classes_):
    fpr, tpr, thresholds = roc_curve(y_test, proba[:, i], pos_label=labels)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=labels, linewidth=2)
    print("AUC for "+labels+" = "+str(roc_auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(b=True)
plt.legend()
plt.savefig("ROC_3class_pipeline.png",dpi=300,transparent=True)

# %% Cross-validation intro

from sklearn.model_selection import cross_val_score
from numpy import arange

mdl = make_pipeline(StandardScaler(),
                    PCA(n_components=50),
                    OneVsRestClassifier(SVC(kernel="poly",
                                            degree=3,
                                            probability=True)))

scores = cross_val_score(mdl, df_seg, df_info["Instrument"], cv=5)

plt.figure()
plt.bar(arange(1,len(scores)+1), scores)
plt.ylabel("Score")
plt.grid(b=True)
plt.savefig("score_crossVal_5fold.png",dpi=300,transparent=True)

# %% 3-class, pipeline with StandardScaler, PCA, SVM. KFold cross-val with ROC

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import numpy as np

#cv = KFold(n_splits=5, shuffle=True)
cv = StratifiedKFold(n_splits=5, shuffle=True)
mdl = make_pipeline(StandardScaler(),
                    PCA(n_components=50),
                    OneVsRestClassifier(SVC(kernel="poly",
                                            degree=3,
                                            probability=True)))

def GetROC(X, y, FigTitle=None):
    
    tprs = []
    #fprs = []
    aucs = []
    Tpr=[]
    Fpr=[]
    #Auc = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure()
    i = 0
    for train, test in cv.split(X, y):
        probas_ = mdl.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test],
                                         probas_[:, 0],
                                         pos_label="Piano")
        tprs.append(interp(mean_fpr, fpr, tpr))
        Tpr.append(tpr)
        Fpr.append(fpr)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    plt.fill_between(mean_fpr,
                     tprs_lower,
                     tprs_upper,
                     color='grey',
                     alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(FigTitle)
    plt.legend(loc="lower right")
    plt.grid(b=True)
    plt.show()
    
    
    D = dict({'mean_tpr': mean_tpr,
             'mean_fpr':mean_fpr,
             'mean_auc':mean_auc,
             'std_auc': std_auc,
             'AllTPR':Tpr,
             'AllFPR':Fpr,
             'AllAUC': aucs})
    
    return D

ROCdata = GetROC(df_seg, df_info["Instrument"], "ROC for Piano")
plt.savefig('ROC_CrossVal_kfold.png', dpi=300, transparent=True)

# %% Clustering in a non-mixed dataset with GMM, different number of clusters

from sklearn.mixture import GaussianMixture

max_clusters = 10

ix = df_info["Instrument"] == "Violin"

pl = make_pipeline(StandardScaler(),
                    PCA(n_components=2))
new_data = pl.fit(df_seg[ix]).transform(df_seg[ix])

bic = []
for i in range(max_clusters):
    mdl = GaussianMixture(n_components = i+1, covariance_type="full")
    mdl.fit(new_data)
    bic.append(mdl.bic(new_data))
    
    # plot the clustering
    x = np.linspace(-50, 100)
    y = np.linspace(-50, 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = mdl.score_samples(XX)
    Z = Z.reshape(X.shape)
    zp = np.exp(Z)
    
    plt.figure()
    cmap = plt.cm.winter
    CS = plt.contour(X, Y, zp, levels=np.linspace(1e-6,zp.max(),10),cmap=cmap)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    
    plt.scatter(new_data[:, 0], new_data[:, 1],5,'r')
    
    plt.title(str(i+1)+ " clusters")
    plt.savefig("GMM_"+str(i+1)+".png", dpi=300, transparent=True)
    
plt.figure()
plt.plot(np.arange(1,len(bic)+1,1), bic, '-o')
plt.ylabel("BIC")
plt.xlabel("num clusters")
plt.grid(b=True)
plt.savefig('GMM_BIC.png', dpi=300, transparent=True)

# EXERCISE:
# Assign a cluster to each data member; what do they represent?


# %% evaluate several different methods with 2 clusters
from sklearn.cluster import AgglomerativeClustering,Birch,DBSCAN,KMeans,SpectralClustering

ix = df_info["Instrument"] == "Violin"

pl = make_pipeline(StandardScaler(),
                    PCA(n_components=3))
new_data = pl.fit(df_seg[ix]).transform(df_seg[ix])

methods = [AgglomerativeClustering(n_clusters=2),
           Birch(n_clusters=2),
           DBSCAN(),
           KMeans(n_clusters=2),
           SpectralClustering(n_clusters=2)]

names = ["AgglomerativeClustering",
         "Birch",
         "DBSCAN",
         "KMeans",
         "SpectralClustering"]

for i,m in enumerate(methods):
    pred_cluster = m.fit_predict(new_data)
    
    u_p = np.unique(pred_cluster) # unique cluster indices
    fig = plt.figure()
    ax = Axes3D(fig)
    for c in u_p:
        ix = pred_cluster==c
        ax.scatter(new_data[ix, 0], new_data[ix, 1], new_data[ix, 2])
    ax.set_title(names[i])
    fig.savefig(names[i]+".png",dpi=300,transparent=True)