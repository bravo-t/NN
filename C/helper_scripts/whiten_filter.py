#!/tool/pandora64/bin/python3

import numpy as np
import glob
from scipy import interpolate

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

def sizeUp(X):
    xmin,xmax = 0,X.shape[1]
    ymin,ymax = 0,X.shape[0]
    xgrids = xmax - xmin
    ygrids = ymax - ymin
    #print("DEBUG: xgrids = %d, , ygrids = %d",xgrids,ygrids)
    X_grid = np.linspace(xmin,xmax,xgrids)
    Y_grid = np.linspace(ymin,ymax,ygrids)
    #print("DEBUG: X_grid = "+X_grid+", Y_grid = "+Y_grid)
    x,y = np.meshgrid(X_grid,Y_grid)
    f = interpolate.interp2d(x,y,X,kind='cubic')
    xgrids_new = 2 * xgrids - 1
    ygrids_new = 2 * ygrids - 1
    Xnew = np.linspace(xmin,xmax,xgrids_new)
    Ynew = np.linspace(ymin,ymax,ygrids_new)
    out = f(Xnew,Ynew)
    return out

files = glob.glob("img/*.pgm")
for f in files:
    print("Processing "+f)
    data = readPGM(f)
    white = whitenData(data)
    white = sizeUp(sizeUp(white))
    max_value = np.max(white)
    min_value = np.min(white)
    if max_value == min_value:
        continue
    white = white - min_value
    m = 255/(max_value - min_value)
    white = white * m
    temp = f.split("/")
    filename = temp[1]
    writePGM(white,"img_white/"+filename)


