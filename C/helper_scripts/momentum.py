from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def func(x,y):
  return (0.3*y**3+y**2+0.3*x**3+x**2)

def func_grad(x,y):
  return (0.9*x**2+2*x, 0.9*y**2+2*y )

def plot_func(xt,yt,c='r'):
  fig = plt.figure()
  ax = fig.gca(projection='3d',
        elev=7., azim=-175)
  X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
  Z = func(X,Y) 
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
    cmap=cm.coolwarm, linewidth=0.1, alpha=0.3)
  ax.set_zlim(-20, 100)
  ax.scatter(xt, yt, func(xt,yt),c=c, marker='o' )
  ax.set_title("x=%.5f, y=%.5f, f(x,y)=%.5f"%(xt,yt,func(xt,yt))) 
  plt.show()
  plt.close()

def run_grad():
  xt = 3 
  yt = 3 
  eta = 0.1
  plot_func(xt,yt,'r')
  for i in range(20):
    gxt, gyt = func_grad(xt,yt)
    xt = xt - eta * gxt
    yt = yt - eta * gyt
    if xt < -5 or yt < -5 or xt > 5 or yt > 5:
      break
    plot_func(xt,yt,'r')

def run_momentum():
  xt = 3 
  yt = 3 
  eta = 0.2
  beta = 0.9
  plot_func(xt,yt,'b')
  delta_x = 0
  delta_y = 0
  for i in range(20):
    gxt, gyt = func_grad(xt,yt)
    delta_x = beta * delta_x + (1-beta)*eta*gxt
    delta_y = beta * delta_y + (1-beta)*eta*gyt
    xt = xt - delta_x
    yt = yt - delta_y 
    if xt < -5 or yt < -5 or xt > 5 or yt > 5:
      break
    plot_func(xt,yt,'b')