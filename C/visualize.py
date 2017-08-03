#!env python3
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
 
# Size our weights arrays
n = 256
 
W = []
for i in range(2):
    W.append(np.linspace(-5,5,n))
 
# Define our data points
Input_Data = [(2,), (0,), (2,), (2.1,)]
Output_Data = [(0.95,), (0.5,), (0.10,), (0.099,)]
 
# Define our supporting functions
 
def sgm(x):
    return 1.0 / (1.0 + np.exp(-x))
 
def E(x,y):
    err = 0
    for i,In in enumerate(Input_Data):
        output = sgm(x*In[0] + y)
        err += 0.5*(output - Output_Data[i][0])**2
 
    return err
 
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
 
# Define the derivative functions
def b_deriv(Weight, Bias, Input, Target):
    O = sgm(Weight * Input + Bias)
    D = (O-Target)*O*(1-O)
    return D
 
def w_deriv(Weight, Bias, Input, Target):
    return b_deriv(Weight, Bias, Input, Target) * Input
 
# Initial Conditions for each algorithm
Base = [(2,4,0,0)]
 
Base_m = Base[:]
Base_m2 = Base[:]
Base_r = Base[:]
Base_rp = Base[:]
Base_rm = Base[:]
Base_irp = Base[:]
Base_irm = Base[:]
Base_rms = Base[:]
Base_am = Base[:]
 
# Meta-parameters for the algorithms
N_Max = 5000
Error_Max = 0.25
 
Eta = 0.2           # Learning Rate
coeff_m = 0.9       # Momentum Coefficient
 
Eta_plus = 1.2      # RProp Factors
Eta_minus = 0.5
lr_min = 1e-6
lr_max = 50
 
lnRMS = 0.9         # RMSProp Factor
 
# Which learnings methods are we running?
llGradient = True
llRMSProp = False
 
llMomentum1 = True
llMomentum2 = False
llAdaptiveMomentum = False
 
llRProp = True
llRProp_p = False
llRProp_m = False
llRProp_ip = True
llRProp_im = False
 
# Iterate and build the solution steps (Gradient descent)
if llGradient:
    print("Gradient Descent...")
    for index in range(N_Max):
        # Break early if error is small
        if E(Base[-1][0] + Base[-1][2], Base[-1][1] + Base[-1][3]) < Error_Max:
            print(("\tIter: {0}".format(index+1)))
            break
 
        if index == N_Max - 1:
            print(("\tIter: {0}".format(index+1)))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base[-1][0] + Base[-1][2], Base[-1][1] + Base[-1][3]
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        # Mark the offset
        Base.append( (w, b, Eta*dw, Eta*db) )
 
# Iterate and build the solution steps (Gradient + Momentum)
if llMomentum1:
    print("Gradient Descent + Momentum...")
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_m[-1][0] + Base_m[-1][2], Base_m[-1][1] + Base_m[-1][3]) < Error_Max:
            print(("\tIter: {0}".format(index+1)))
            break
 
        if index == N_Max - 1:
            print(("\tIter: {0}".format(index+1)))
            break
 
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_m[-1][0] + Base_m[-1][2], Base_m[-1][1] + Base_m[-1][3]
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        # Mark the offset
        Base_m.append( (w, b, coeff_m * Base_m[-1][2] + Eta*dw, coeff_m * Base_m[-1][3] + Eta*db) )
 
# Iterate and build the solution steps (Nesterov / Sutskever)
if llMomentum2:
    print("Gradient Descent + (Nesterov / Sutskever) Momentum...")
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_m2[-1][0] + Base_m2[-1][2], Base_m2[-1][1] + Base_m2[-1][3]) < Error_Max:
            print(("\tIter: {0}".format(index+1)))
            break
 
        if index == N_Max - 1:
            print(("\tIter: {0}".format(index+1)))
            break
 
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_m2[-1][0] + Base_m2[-1][2], Base_m2[-1][1] + Base_m2[-1][3]
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w + coeff_m * Base_m2[-1][2], b + coeff_m * Base_m2[-1][3], Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w + coeff_m * Base_m2[-1][2], b + coeff_m * Base_m2[-1][3], Input_Data[i][0], Output_Data[i][0])
 
        # Mark the offset
        Base_m2.append( (w, b, coeff_m * Base_m2[-1][2] + Eta*dw, coeff_m * Base_m2[-1][3] + Eta*db) )
 
# Iterate and build the solution steps (Gradient + Adaptive Momentum)
if llAdaptiveMomentum:
    print("Gradient Descent + Adaptive Momentum...")
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_am[-1][0] + Base_am[-1][2], Base_am[-1][1] + Base_am[-1][3]) < Error_Max:
            print(("\tIter: {0}".format(index+1)))
            break
 
        if index == N_Max - 1:
            print(("\tIter: {0}".format(index+1)))
            break
 
        # Compute the derivatives
        dw, db = 0, 0
        pw, pb = Base_am[-1][2], Base_am[-1][3]
        w, b = Base_am[-1][0] + pw, Base_am[-1][1] + pb
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
         
        P = np.array([pw, pb], dtype = 'float32')
        D = np.array([dw, db], dtype = 'float32')
         
        lP, lD = np.sqrt(np.dot(P,P)), np.sqrt(np.dot(D,D))
        if lP != 0 and lD != 0:
            c_m = (1 + (np.dot(P,D) / np.sqrt(np.dot(P,P)*np.dot(D,D))))**2 / 4
        else:
            c_m = 0.05
         
        # Mark the offset
        Base_am.append( (w, b, c_m * Base_am[-1][2] + Eta*dw, c_m * Base_am[-1][3] + Eta*db) )
 
# Iterate and build the solution steps (RProp)
if llRProp:
    print("RProp...")
    lr = [0.1, 0.1]
    prev = [0, 0]
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_r[-1][0] + Base_r[-1][2], Base_r[-1][1] + Base_r[-1][3]) < Error_Max:
            print("\tIter: {0}".format(index+1))
            break
 
        if index == N_Max - 1:
            print("\tIter: {0}".format(index+1))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_r[-1][0] + Base_r[-1][2], Base_r[-1][1] + Base_r[-1][3]
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        curr = [dw, db]
        delta = [0, 0]
        # Compute the rprop algorithm
        for i in range(len(curr)):
            if curr[i] * prev[i] > 0:
                lr[i] = min([lr[i] * Eta_plus, lr_max])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
                 
            elif curr[i] * prev[i] < 0:
                lr[i] = max([lr[i] * Eta_minus, lr_min])
                prev[i] = 0
                #delta[i] = 0
 
            else:
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
                 
        # Mark the offset
        Base_r.append( (w, b, delta[0], delta[1]) )
 
# Iterate and build the solution steps (RProp+)
if llRProp_p:
    print("RProp+...")
    lr = [0.1, 0.1]
    prev = [0, 0]
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_rp[-1][0] + Base_rp[-1][2], Base_rp[-1][1] + Base_rp[-1][3]) < Error_Max:
            print("\tIter: {0}".format(index+1))
            break
 
        if index == N_Max - 1:
            print("\tIter: {0}".format(index+1))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_rp[-1][0] + Base_rp[-1][2], Base_rp[-1][1] + Base_rp[-1][3]
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        curr = [dw, db]
        delta = [0, 0]
        # Compute the rprop+ algorithm
        for i in range(len(curr)):
            if curr[i] * prev[i] > 0:
                lr[i] = min([lr[i] * Eta_plus, lr_max])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
                 
            elif curr[i] * prev[i] < 0:
                lr[i] = max([lr[i] * Eta_minus, lr_min])
                delta[i] = -Base_rp[-1][2+i]
                prev[i] = 0
 
            else:
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
         
        # Mark the offset
        Base_rp.append( (w, b, delta[0], delta[1]) )
 
# Iterate and build the solution steps (RProp-)
if llRProp_m:
    print("RProp-...")
    lr = [0.1, 0.1]
    prev = [0, 0]
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_rm[-1][0] + Base_rm[-1][2], Base_rm[-1][1] + Base_rm[-1][3]) < Error_Max:
            print("\tIter: {0}".format(index+1))
            break
 
        if index == N_Max - 1:
            print("\tIter: {0}".format(index+1))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_rm[-1][0] + Base_rm[-1][2], Base_rm[-1][1] + Base_rm[-1][3]
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        curr = [dw, db]
        delta = [0, 0]
        # Compute the rprop- algorithm
        for i in range(len(curr)):
            if curr[i] * prev[i] > 0:
                lr[i] = min([lr[i] * Eta_plus, lr_max])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
                 
            elif curr[i] * prev[i] < 0:
                lr[i] = max([lr[i] * Eta_minus, lr_min])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
 
            else:
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
         
        # Mark the offset
        Base_rm.append( (w, b, delta[0], delta[1]) )
 
# Iterate and build the solution steps (iRProp+)
if llRProp_ip:
    print("iRProp+...")
    lr = [0.1, 0.1]
    prev = [0, 0]
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_irp[-1][0] + Base_irp[-1][2], Base_irp[-1][1] + Base_irp[-1][3]) < Error_Max:
            print("\tIter: {0}".format(index+1))
            break
 
        if index == N_Max - 1:
            print("\tIter: {0}".format(index+1))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_irp[-1][0] + Base_irp[-1][2], Base_irp[-1][1] + Base_irp[-1][3]
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        curr = [dw, db]
        delta = [0, 0]
        # Compute the irprop+ algorithm
        for i in range(len(curr)):
            if curr[i] * prev[i] > 0:
                lr[i] = min([lr[i] * Eta_plus, lr_max])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
                 
            elif curr[i] * prev[i] < 0:
                lr[i] = max([lr[i] * Eta_minus, lr_min])
                if E(w, b) > E(Base_irp[-1][0], Base_irp[-1][1]):
                    delta[i] = -Base_irp[-1][2+i]
                prev[i] = 0
 
            else:
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
         
        # Mark the offset
        Base_irp.append( (w, b, delta[0], delta[1]) )
 
# Iterate and build the solution steps (iRProp-)
if llRProp_im:
    print("iRProp-...")
    lr = [0.1, 0.1]
    prev = [0, 0]
    for index in range(N_Max):
        # Break early if error is small
        if E(Base_irm[-1][0] + Base_irm[-1][2], Base_irm[-1][1] + Base_irm[-1][3]) < Error_Max:
            print("\tIter: {0}".format(index+1))
            break
 
        if index == N_Max - 1:
            print("\tIter: {0}".format(index+1))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
        w, b = Base_irm[-1][0] + Base_irm[-1][2], Base_irm[-1][1] + Base_irm[-1][3]
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        curr = [dw, db]
        delta = [0, 0]
        # Compute the irprop- algorithm
        for i in range(len(curr)):
            if curr[i] * prev[i] > 0:
                lr[i] = min([lr[i] * Eta_plus, lr_max])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
                 
            elif curr[i] * prev[i] < 0:
                lr[i] = max([lr[i] * Eta_minus, lr_min])
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = 0
 
            else:
                delta[i] = sign(curr[i]) * lr[i]
                prev[i] = curr[i]
         
        # Mark the offset
        Base_irm.append( (w, b, delta[0], delta[1]) )
 
# Iterate and build the solution steps (RMSProp)
MS_History = []
if llRMSProp:
    print("RMSProp...")
    MS = np.array([0, 0],dtype='float32')
    for index in range(N_Max):
        # Break early if error is small
        w, b = Base_rms[-1][0] + Base_rms[-1][2], Base_rms[-1][1] + Base_rms[-1][3]
        Err = E(w, b)
        if Err < Error_Max:
            print("\tIter: {0}".format(index+1))
            break
 
        if index == N_Max - 1:
            print("\tIter: {0}".format(index+1))
            break
         
        # Compute the derivatives
        dw, db = 0, 0
         
        for i in range(len(Input_Data)):
            dw -= w_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
            db -= b_deriv(w, b, Input_Data[i][0], Output_Data[i][0])
 
        # Mark the offset
        D = np.array([dw, db])
        MS = lnRMS * MS + (1 - lnRMS) * D**2
        MS_History.append(MS)
         
        sMS = np.sqrt(MS)
        Base_rms.append( (w, b, Eta*dw/sMS[0], Eta*db/sMS[1]) )
 
#
# Build a contour plot
#
X, Y = np.meshgrid(W[0], W[1])
Z = E(X, Y)
 
plt.figure()
 
CSF = plt.contourf(X, Y, Z, 14, alpha = .3, cmap = cm.jet,zorder = 0)
CS = plt.contour(X, Y, Z,14, colors = 'black', zorder = 1)
 
# Plot Gradient Descent
if len(Base) > 1:
    plt.quiver(*list(zip(*Base)),
                color = 'red',width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 2)
 
# Plot Gradient Descent with Momentum 1
if len(Base_m) > 1:
    plt.quiver(*list(zip(*Base_m)),
                color = 'blue',width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 3)
 
# Plot Gradient Descent with Momentum 2
if len(Base_m2) > 1:
    plt.quiver(*list(zip(*Base_m2)),
                color = 'green',width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 4)
 
# Plot RProp
if len(Base_r) > 1:
    plt.quiver(*list(zip(*Base_r)),
                color = (1.0,0.8,0.8),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 5)
 
# Plot RProp+
if len(Base_rp) > 1:
    plt.quiver(*list(zip(*Base_rp)),
                color = (0.5,0,1),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 6)
 
# Plot RProp-
if len(Base_rm) > 1:
    plt.quiver(*list(zip(*Base_rm)),
                color = (0,1,1),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 6)
 
# Plot iRProp+
if len(Base_irp) > 1:
    plt.quiver(*list(zip(*Base_irp)),
                color = (0.1,0.1,0.5),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 7)
 
# Plot iRProp-
if len(Base_irm) > 1:
    plt.quiver(*list(zip(*Base_irm)),
                color = (0.1,0.5,0.5),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 8)
 
# Plot RMSProp
if len(Base_rms) > 1:
    plt.quiver(*list(zip(*Base_rms)),
                color = (0.8,0.8,0.2),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 9)
 
# Plot Adaptive Momentum
if len(Base_am) > 1:
    plt.quiver(*list(zip(*Base_am)),
                color = (0.1,0.65,0.65),width = 0.005,
                scale_units = 'xy', scale = 1.0,
                angles = 'xy', zorder = 10)
 
plt.clabel(CS, inline = 1, fontsize = 10)
plt.title('Error function in weight-space')
 
plt.show()