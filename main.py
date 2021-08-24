import pandas as pd
import numpy as np

from sys import argv


def mahalanobis(x:np.ndarray, data:np.ndarray):
    x_mu = x - np.mean(data)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

def main():
    try:
        file = argv[1]
    except IndexError:
        print("enter valid path for .csv analysis")
        exit()

    data = pd.read_csv(file, sep="\t")
    data["division"] = data["division"].astype(np.uint64)
    data["year"] = data["year"].astype(np.uint16)
    data["nclass"] = data["nclass"].astype(np.uint8)
    data["firm"] = data["firm"].astype(np.uint16)

    data = data.sort_values("nclass")
    tclass_replacements = dict((k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique())))
    data["tclass"] = data.nclass.replace(tclass_replacements)

    data = data.sort_values(["firm", "tclass"])

    subsh = pd.crosstab(data["firm"], data["tclass"]).astype(np.uint32)
    total = subsh.values.sum(axis=1)
    subsh = (subsh.values / total[:, None]) * 100

    del total

    var = np.dot(subsh.T, subsh)
    base_var = var.copy()

    for i in range(var.shape[0]):
        for j in range(var.shape[0]):
            var[i,j] = var[i,j] / (np.sqrt(base_var[i,i]) * np.sqrt(base_var[j,j]))

    norm_subsh = subsh.copy()
    base_std = np.dot(subsh, subsh.T)
    for i in range(base_std.shape[0]):
        for j in range(var.shape[0]):
            norm_subsh[i,j] = subsh[i,j] / np.sqrt(base_std[i,i])
    
    std = np.dot(norm_subsh, norm_subsh.T)
    cov_std = np.dot(subsh, subsh.T)

    mal_corr = np.dot(np.dot(norm_subsh, var),norm_subsh.T)
    standard = mal_corr.copy()
    covmal_corr = np.dot(np.dot(subsh, var), subsh.T)
    cov_standard = covmal_corr.copy()

    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()