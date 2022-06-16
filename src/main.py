import numpy as np
import os
import typing
import h5py

import cv2
import scipy.io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import sklearn as sk
import joblib

from pathlib import Path
from os import listdir
from os.path import isfile, join

import sklearn.metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB, CategoricalNB


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve,classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, balanced_accuracy_score, label_ranking_average_precision_score

import scipy.io

from sklearn.decomposition import PCA, KernelPCA

# Where to save the figures
PROJECT_ROOT_DIR = "/home/bret/PycharmProjects/623-Project"
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def decision_tree(x_data,y_data):
    clf = DecisionTreeClassifier(random_state=42)
    model = 0
    return model

def KNN(x_data,y_data):

    return

def logistic_regression(x_data,y_data): #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = LogisticRegression(random_state=0).fit(x_data,y_data)
    return clf

def SVM(x_data,y_data):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_data, y_data)

    return clf

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)





def get_data(pris_file, mat_file ):
    """ Gathers Data from OPT"""
    Data = np.zeros([1000,1000,172])
    Train_Data = np.zeros([1000,750,172]) #Sets up the numpy array for the big reflectance cubes
    Test_Data = np.zeros([1000,200,172]) #Sets up the numpy array for the big reflectance cubes
    f = h5py.File(pris_file)
    # reading VNIR datacube; adjust the “PRS_L1_HCO” string portion depending by the specific PRISMA product type (e.g. for L2D product use PRS_L2D_HCO)
    cubeDataSWIR = f['/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube']
    minSWIR= f.attrs['L2ScaleSwirMin']
    maxSWIR = f.attrs['L2ScaleSwirMax']
    reflCubeSWIR= minSWIR + (cubeDataSWIR) * (maxSWIR - minSWIR) / 65535
    X_Data = reflCubeSWIR[:, 0:171, :]
    mat_data = scipy.io.loadmat(mat_file)
    X_Data = np.transpose(X_Data, (0,2,1))
    Data[:,:,0:171] = X_Data
    Data[:,:,-1] = mat_data['ClassMap']
    Train_Data = Data[:,0:750,:]
    Test_Data = Data[:,800:1000,:]
    Train_Data = np.reshape(Train_Data,(750000,172))
    Test_Data = np.reshape(Test_Data,(200000,172))


    return Train_Data, Test_Data



def main():
    print(f'Final Project')
    MakeDataframe = False
    Explore = False
    Train = "Small"
    Model = 'SVC_LIN' #LDA, QDA, SVC_RBF, SVC_LIN, GNB, DUM, DT, LASSO
    Model_Name = Model + '.pkl'
    Fit_Model = False
    Evaluate_Model = True
    random_state = 42
    Eval_Data = 'Val' #Train Val Test
    Playground = False #A function for testing different items & exploring dataframe
    Class_labels = ['Water', 'Trees','Grass','Flooded', 'Crops', 'Scrub', 'Build', 'Bare', 'Snow']


    print("Grabbing Data ")
    # Declare all of the files used for training
    # Mat files are the Classmaps Generated in Matlab (Y data)
    # Pris files are the Sattelite Data (X_Data)
    mat1 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/ClassMap_PRS_L2C_STD_20200614180702_20200614180706_0001.mat' # Northern CO
    pris1 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/PRS_L2C_STD_20200614180702_20200614180706_0001.he5' #northern CO
    mat2 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/ClassMap_PRS_L2C_STD_20210703164629_20210703164633_0001.mat' #northern IN
    pris2 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/PRS_L2C_STD_20210703164629_20210703164633_0001.he5' #northern IN
    mat3 = '/home/bret/Downloads/OneDrive_2_6-1-2022/ClassMap_PRS_L2C_STD_20210925190934_20210925190938_0001.mat' # Mt Raineer
    pris3 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/PRS_L2C_STD_20210925190934_20210925190938_0001.he5' # Mt Raineer
    mat4 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/ClassMap_PRS_L2C_STD_20220212161700_20220212161705_0001.mat' # Cape Canavral
    pris4 = '/home/bret/Documents/MATLAB/ASSET/Prisma/PRISMA_Data/PRS_L2C_STD_20220212161700_20220212161705_0001.he5' # Cape Canavral


    if MakeDataframe == True:
        print("Making Dataframe")
        train, test = get_data(pris1,mat1)
        train2, test2 = get_data(pris2, mat2)
        train3, test3 = get_data(pris3, mat3)
        train4, test4 = get_data(pris4, mat4)
        df_train = pd.DataFrame(train)
        df_test = pd.DataFrame(test)
        df_train2 = pd.DataFrame(train2)
        df_test2 = pd.DataFrame(test2)
        df_train3 = pd.DataFrame(train3)
        df_test3 = pd.DataFrame(test3)
        df_train4 = pd.DataFrame(train4)
        df_test4 = pd.DataFrame(test4)

        #Put all the files together
        df_train = pd.concat([df_train,df_train2,df_train3, df_train4])
        df_test = pd.concat([df_test,df_test2,df_test3, df_test4])
        df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df_train = df_train.rename(columns={171: 'Class'})
        df_test = df_test.rename(columns={171: 'Class'})
        df_train = df_train[df_train.Class != 0.0] #Drop No-Data -
        df_test = df_test[df_test.Class != 0.0]
        df_train = df_train[df_train.Class != 10.0] #Drop Cloud Cover 10
        df_test = df_test[df_test.Class != 10.0]
        df_train = df_train[df_train.Class != 11.0] #Drop No-Data 11
        df_test = df_test[df_test.Class != 11.0]
        df_train.info()
        df_train.to_pickle('train_df')
        df_test.to_pickle('test_df')
        if Train == 'Small':
            df_train = df_train.groupby('Class')
            df_train = pd.DataFrame(df_train.apply(lambda x: x.sample(df_train.size().min()).reset_index(drop=True)))
            df_train.info()
            counts = df_train.Class
            print(counts.value_counts())
            df_train.to_pickle('train_df_small')

    else:
        print('Loading Dataframe')
        if Train == "Small":
            df_train = pd.read_pickle('train_df_small')
        else:
            df_train = pd.read_pickle('train_df')
        df_test = pd.read_pickle('test_df')

    #Explore the training dataset
    if Explore == True:
        df_train.info()
        df_train.describe()
        Classes = df_test['Class']
        Classes.hist(bins=9, figsize=(20, 15))
        save_fig("collumn_histogram_plots")
        Classes.value_counts()
        print(Classes.value_counts())
        plt.show()  #  df_train.columns()
        Corr_Matrix = df_train.corr()
        Corr_Matrix.to_pickle("Corr_Matrix")
        Var_Matrix = df_train[0:171].var()
        Var_Matrix.to_pickle("Var_Matrix") #np.min(Var_Matrix[0:170]) 4.127963872508656e-06 np.max(Var_Matrix[0:170])  0.008223567159650523 np.mean(Var_Matrix[0:170]) 0.003690186590632295


    # Playground Feature to test out different data features
    if Playground == True:
        Var_Matrix = pd.read_pickle("Var_Matrix")
        Corr_Matrix = pd.read_pickle("Corr_Matrix")
        Band = 2
        Corr_band = Corr_Matrix[Band]
        Corr_band = Corr_band[0:170]
        plt.plot(Corr_band)
        plt.title("Correlation to band "+ str(Band))
        plt.xlabel("Band")
        save_fig("Correlation "+ str(Band))




    #Split Data Frames into X & Y Data
    x_train = df_train[np.arange(0,170)]
    y_train = df_train['Class']
    x_test = df_test[np.arange(0,170)]
    y_test = df_test['Class']



    scaler = StandardScaler() # Z scaler seemed to make no difference for LDA
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components='mle') #tested with values 10, 40, 100, 160, 'MLE' 170; Accuraccy Only Declines with LDA
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)


    skf = StratifiedKFold(n_splits=5)


    if Fit_Model == True:
        print('Fitting Model: ', Model)
        if Model == 'LDA': #54.28% Crossval
            LDA_Model = LinearDiscriminantAnalysis()
            Chosen_Model = LDA_Model
        elif Model == 'QDA': #57.056 Crossval
            QDA_Model = QuadraticDiscriminantAnalysis()
            Chosen_Model = QDA_Model
        elif Model == 'SVC_RBF': #42% Crossval
            SVC_RBF_Model = SVC(gamma='auto')
            SVC_RBF_Model.fit(x_train,y_train)
            Chosen_Model = SVC_RBF_Model
        elif Model == 'SVC_LIN': #58.419539178071446%
            SVC_LIN_Model = LinearSVC(random_state=42, C=0.6)
            # SVC_LIN_Model.fit(x_train,y_train)
            # parameters = { 'C':[.4,.5,.6]} #.01,.05,.1,.5,1,5  search done again on .4 .5 .6 found .6 best by a small margin.
            # clf = GridSearchCV(SVC_LIN_Model, parameters)
            # clf.fit(x_train,y_train)
            # best_est = clf.best_estimator_
            # best_parms = clf.best_params_
            # scores = clf.cv_results_['mean_test_score']#[0.57126771, 0.58263282, 0.58546974, 0.58670929, 0.58085208, 0.48297408]
            # scores = cross_val_score(clf, x_train_pca, y_train, cv=5)
            # print(scores)
            #Mean fit time 6 minutes 1 second
            Chosen_Model = SVC_LIN_Model
        elif Model == 'GNB':
            GNB_Model = GaussianNB()
           # GNB_Model.fit(x_train,y_train)
            Chosen_Model = GNB_Model
        elif Model == 'KNN':
            KNN_Model = KNeighborsClassifier(5)
            KNN_Model.fit(x_train,y_train)
            Chosen_Model = KNN_Model
        elif Model == 'DT':
            DT_Model = DecisionTreeClassifier()
            DT_Model.fit(x_train,y_train)
            Chosen_Model = DT_Model
        # save trained model
        cv_results = cross_validate(Chosen_Model, x_train_pca, y_train, cv=5)
        score = cv_results['test_score']
        Chosen_Model.fit(x_train,y_train)
        print('Mean CrossVal Accuracy: ' , np.mean(score)*100,'%')
        joblib.dump(Chosen_Model, Model_Name)
    else:
        # load an already trained model
        print('Loading Model')
        Chosen_Model = joblib.load(Model_Name)

    # cv_results = cross_validate(Chosen_Model, x_train, y_train, cv=5)
    # score = np.mean(cv_results['test_score'])
    # print(score)




    if Eval_Data == 'Train':
        x_dat, y_dat = x_train, y_train
    elif Eval_Data == 'Val':
        df_val = pd.read_pickle('train_df')
        df_val = df_val.sample(frac=.1, random_state=random_state)
        x_val = df_val[np.arange(0, 170)]
        x_val = scaler.transform(x_val)
        y_val = df_val['Class']
        x_dat, y_dat = x_val, y_val
    elif Eval_Data == 'Test':
        x_dat, y_dat = x_test, y_test


    if Evaluate_Model == True:
        print("Evaluating model on the ", Eval_Data, 'Set')
        print('Chosen Model is: ', Model)
        y_pred = Chosen_Model.predict(x_dat)
        precision, recall, f1_score, support = precision_recall_fscore_support(y_dat,y_pred)
        Chosen_Accuracy = balanced_accuracy_score(y_dat,y_pred)
        class_rep = classification_report(y_dat, y_pred)
        print('Precision: ' , str(precision), '\n', 'Recall: ' , str(recall) ,'\n', 'F1 Score: ' , str(f1_score))
        print('Balanced Accuracy: ', Chosen_Accuracy)
        #LDA: QDA:.577
        cm = sklearn.metrics.confusion_matrix(y_dat, y_pred)
        disp = sklearn.metrics.ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()
        class_acc = cm.diagonal()/cm.sum(axis=1)
        print('Class Accuracy: ', class_acc)


        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(Class_labels)
        ax.yaxis.set_ticklabels(Class_labels)
        plt.show()
        save_fig(Model + " Confusion Matrix", ax)

        print(class_acc)


    print("Sucess!")


if __name__ == '__main__':
    main()
