import numpy as np
import pandas as pd

#Return the output data of the first stage learning, without labels
def get_X():
    X_ecologie = np.loadtxt("X_ecologie.csv", delimiter=",")
    X_democratie = np.loadtxt("X_democratie.csv", delimiter=",")
    X_fiscalite = np.loadtxt("X_fiscalite.csv", delimiter=",")
    X_organisation = np.loadtxt("X_organisation.csv", delimiter=",")
    X = np.concatenate((X_ecologie, X_democratie, X_fiscalite, X_organisation), axis=1)
    return(X)


#Return the output data of the first stage learning with labels
def get_full_X():
    X_ecologie = np.loadtxt("X_ecologie.csv", delimiter=",")
    X_democratie = np.loadtxt("X_democratie.csv", delimiter=",")
    X_fiscalite = np.loadtxt("X_fiscalite.csv", delimiter=",")
    X_organisation = np.loadtxt("X_organisation.csv", delimiter=",")

    X = np.concatenate((X_ecologie, X_democratie, X_fiscalite, X_organisation), axis=1)

    columns = ["ecologie_label"+str(i) for i in range(len(X_ecologie[0]))]
    columns = columns + ["democratie_label"+str(i) for i in range(len(X_democratie[0]))]
    columns = columns + ["fiscalite_label"+str(i) for i in range(len(X_fiscalite[0]))]
    columns = columns + ["organisation_label"+str(i) for i in range(len(X_organisation[0]))]
    columns = [""] + columns
    columns = np.array(columns).reshape(1, -1)

    four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
    rows = np.array(four_surveys_taken_auth_ids).reshape(-1,1)
    X = np.concatenate((rows, X), axis=1)
    X = np.concatenate((columns, X), axis=0)

    X = pd.DataFrame(data=X[1:,1:], index=X[1:,0], columns=X[0,1:])
    return(X)

#Return the labels of X
def get_labels():
    X_ecologie = np.loadtxt("X_ecologie.csv", delimiter=",")
    X_democratie = np.loadtxt("X_democratie.csv", delimiter=",")
    X_fiscalite = np.loadtxt("X_fiscalite.csv", delimiter=",")
    X_organisation = np.loadtxt("X_organisation.csv", delimiter=",")

    columns = ["ecologie_label"+str(i) for i in range(len(X_ecologie[0]))]
    columns = columns + ["democratie_label"+str(i) for i in range(len(X_democratie[0]))]
    columns = columns + ["fiscalite_label"+str(i) for i in range(len(X_fiscalite[0]))]
    columns = columns + ["organisation_label"+str(i) for i in range(len(X_organisation[0]))]
    #columns = [""] + columns
    #columns = np.array(columns).reshape(1, -1)
    return(columns)

#Return the ids of the author who aswered the 4 themes
def get_auth_id():
    return(np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str))   

