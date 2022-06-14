# generic import
import os
import time
import ftplib
import datetime
import argparse
import numpy as np

from pyriemann.classification import FgMDM, MDM, TSclassifier
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance
# sklearn imports
from sklearn.model_selection import cross_val_score, KFold

from datasetEEG import DatasetAffectRobot, DataLoaderAffectRobot


def learn_barycenter(data_loader):
    cov_train = data_loader._spd_mat
    cov_labels = data_loader._spd_labels
    sample_weight = np.ones(cov_train.shape[0])
    covmeans_ = [mean_covariance(cov_train[cov_labels == l + 1], metric='riemann', sample_weight=sample_weight[cov_labels == l+1])
                     for l in range(int(max(cov_labels)))]   # 1 - negative, 2 - positive
    return covmeans_


def predict_distances(barycenters, covtest):
    Nc = len(barycenters)
    dist = [distance(covtest, barycenters[m], 'riemann') for m in range(Nc)]
    dist = np.concatenate(dist, axis=None)
    return dist # len = 2. [negative, positive]


def predict_class(barycenters, covtest):
    dist = predict_distances(barycenters, covtest)
    Nc = len(dist)
    labels = [l for l in range(Nc)]
    return labels[dist.argmin()] + 1 #1 : Negative, #2 - positive



def predict_valence(barycenters, covtest):
    pred_class = predict_class(barycenters, covtest)
    pred_dist = predict_distances(barycenters, covtest) # distance between the two barycenters (positive, negative) and a sample point
    pred_dist_intercenters = predict_distances(barycenters, barycenters[0])[1] # distance between the negative and the positive valence states
    d_n = pred_dist[0]  # distance between the negative center and the query
    d_p = pred_dist[1]  # distance between the positive center and the query
    d_c = pred_dist_intercenters
    cosine_neg = (d_p**2 - d_n**2 - d_c**2) / (-2 * d_n * d_c)
    cosine_pos = (d_n**2 - d_p**2 - d_c**2) / (-2 * d_p * d_c)
    exp_cosine = min(np.abs(cosine_neg), np.abs(cosine_pos)) / max(np.abs(cosine_neg), np.abs(cosine_pos))
    #print(cosine_pos)
    #print(cosine_neg)
    #valence = 3 # min=1, max=6
    if pred_class == 1: # negative
        #print("negative")
        valence = 3.5 - 2.5 * (1 - exp_cosine)
    else:   # positive
        #print("positive")
        valence = 3.5 + 2.5 * (1 - exp_cosine)#valence - 20 * cosine_pos + 1
    #print(valence)
    return valence
    #return [cosine_neg, cosine_pos, exp_dist]


def eval(data_loader, rtype, cv):
    cov_train = data_loader._spd_mat
    cov_labels = data_loader._spd_labels
    if rtype == 'MDM':
        mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    elif rtype == 'FGMDM':
        mdm = FgMDM(metric=dict(mean='riemann', distance='riemann'))
    scores = cross_val_score(mdm, cov_train, cov_labels, cv=cv, n_jobs=1)
    mdm.fit(cov_train, cov_labels)
    # Printing the results
    print(rtype + " Classification accuracy: ", np.mean(scores))
    return mdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Online_main")
    parser.add_argument("-s", "--subject", metavar = '', type = str, required = True, help = "The first name of subject in uppercase")
    parser.add_argument("-d", "--date", metavar = '', type = str, required = True, help = "Today's date (YYMMDD)")
    args = parser.parse_args()
    valence = 0
    # Train
    while True:
        if not os.path.isfile(os.path.join(os.getcwd(), "Subject_Names", args.subject, "Train", args.date, "labels_VA.csv")):
            continue

        start_time = time.time()
        d = DataLoaderAffectRobot(os.path.join(os.getcwd(), "Subject_Names", args.subject, "Train", args.date+"\\"), 5, 0.2, 1, cov=True, train=True).getAllData()
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Data loading Finished. Total elapsed time (h:m:s): {}".format(elapsed))
        # cross validation
        #cv = KFold(n_splits=5)
        #eval(d.dataset, 'MDM', cv)
        #mdm = eval(d.dataset, 'FGMDM', cv)
        global cov_means
        cov_means = learn_barycenter(d.dataset)

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
        break

    filename = "log_"+args.subject+"_"+args.date+"_score.txt"

    idx_obj = 1
    obj_list = ['EMPTY', 'FULL', 'COOKIE']
    while True:
        time.sleep(1)
        if os.listdir(os.path.join(os.getcwd(), "Subject_Names", args.subject, "Test", args.date)) == 0:
            continue
        for obj in obj_list:
            while not os.path.isfile(
                    os.path.join(os.getcwd(), "Subject_Names", args.subject, "Test", args.date, obj,
                                 obj + str(idx_obj),
                                 obj + str(idx_obj) + "_" + args.subject + "_" + args.date + ".npy")):
                continue
            time.sleep(1)
            testData = DataLoaderAffectRobot(
                os.path.join(os.getcwd(), "Subject_Names", args.subject, "Test", args.date, obj,
                             obj + str(idx_obj) + "\\"), 5, 0.2, 1, cov=False, train=False).getAllData()
            #result = mdm.predict(testData.dataset._spd_mat)
            valences = np.concatenate([predict_valence(cov_means, cov_test) for cov_test in testData.dataset._spd_mat], axis = None)
            labels = np.concatenate(testData.dataset._state, axis = None)
            result = []
            #print(cov_means)
            #print(valences.shape)
            #print(labels.shape)
            for sdx in np.unique(testData.dataset._state):
                v_max = max(valences[sdx == labels])
                v_min = min(valences[sdx == labels])
                v_mean = (v_max + v_min) / 2
                if v_mean < 1:
                    v_mean = 1
                elif v_mean > 6:
                    v_mean = 6
                result.append(v_mean)
            #result = [predict_valence(cov_means, cov_test) for cov_test in testData.dataset._spd_mat]
            #print(result)
            sdx = 0
            states = np.unique(testData.dataset._state)
            for rdx in result:
                print("rdx : {}".format(rdx))
                state = states[sdx]#stData.dataset.__getitem__(sdx)[1]

                # print(state)
                list_state = state.split(", ")

                # print('[' + list_state[0] + ", " + list_state[2] + ", " + list_state[3] + ", " + str(rdx) + ']' + "\n")

                ftp = ftplib.FTP()
                ftp.connect("143.248.236.225")
                ftp.login("urp", "kaist2020")
                ftp.cwd("URP")

                f = open(filename, 'a')
                f.write('[' + list_state[0] + ", " + list_state[2] + ", " + list_state[3] + ", " + str(rdx) + ']' + "\n")
                f.close()
                myfile = open(filename, 'rb')
                ftp.storbinary("STOR " + filename, myfile)
                myfile.close()
                ftp.close()
                sdx += 1
        idx_obj += 1