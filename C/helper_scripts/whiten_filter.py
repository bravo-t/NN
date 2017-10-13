#!/tool/pandora64/bin/python3

import numpy as np
import glob

def readPGM(filename):
    with open(filename, 'r') as f:
        f.readline()
        width, height = f.readline().rstrip().split(" ")
        f.readline()
        height = int(float(height))
        width = int(float(width))
        out = np.zeros((height,width))
        for i in range(height):
            row = f.readline().split(' ')
            for j in range(width):
                out[i,j] = row[j]
    return out


def whitenData(X):
    X -=np.mean(X,axis=0)
    cov = np.dot(X.T, X) / X.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X,U)
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    return Xwhite

def writePGM(X,filename):
    with open(filename,'w') as f:
        f.write("P2\n")
        f.write(str(X.shape[1])+" ")
        f.write(str(X.shape[0])+"\n")
        f.write("255\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(str(int(X[i,j]))+" ")
            f.write("\n")

files = glob.glob("img_bk/*.pgm")
for f in files:
    print("Processing "+f)
    data = readPGM(f)
    white = whitenData(data)
    max_value = np.max(white)
    min_value = np.min(white)
    if max_value == min_value:
        continue
    white = white - min_value
    m = 255/(max_value - min_value)
    white = white * m
    temp = f.split("/")
    filename = temp[1]
    writePGM(white,"img/"+filename)


