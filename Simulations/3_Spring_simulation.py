# %% Progam goal
'''
Simulation a 3-DOF mass spring system, using two different integration methods: Euler-Cromer and Beeman. 


by: 
    Diogo Durão 55739
    Mónica Dyreby 55808

'''

# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

#%% Inicialization

def inicialization(size, inicial):
    
    '''
    Goal: initialize the arrays with information about the simulation 
    
    Variables: 
                size: indicates the size of the arrays
                inicial: Array with information regarding the inicial displacements,
                         inicial velocities k values for each spring and m values for each mass 
    
    return:
                t - array which will hold simulation timestep 
                xData - array which will hold simulated positions for all masses 
                vData - array which will hold simulated velocities for all masses 
                EData - array which will hold simulated energies
    '''

    number_of_bodies = len(inicial[0])
    t = np.zeros(size)

    xData = np.zeros((size, number_of_bodies))
    vData = np.zeros((size, number_of_bodies))
    
    xData[0] = inicial[0]
    vData[0] = inicial[1]
    
    EData = np.zeros(size-1)

    return t, xData, vData, EData

#%% Euler-Cromer functions

def EulerCromer_1(t, x, v, a, deltaT, k, m, i):
    
    '''
    Auxiliary funtion of springCalEC(t, x, v, deltaT, k, m, n) for one mass and one spring
    '''        
    
    a[0] = (-k[0]*(x[0]))/m[0]
    
    v[0] = v[0] + a[0]*deltaT
    
    x[0] = x[0] + v[0]*deltaT
    
    

def EulerCromer_2(t, x, v, a, deltaT, k, m, i):
    
    '''
    Auxiliary funtion of springCalEC(t, x, v, deltaT, k, m, n) for two masses and two springs
    '''
    
    a[0] = (-k[0]*x[0])/m[0] + (k[1]*(x[1]-x[0]))/m[0]
    a[1] = -(k[1]*(x[1]-x[0]))/m[1]
    
    v[0] = v[0] + a[0]*deltaT
    v[1] = v[1] + a[1]*deltaT
    
    x[0] = x[0] + v[0]*deltaT
    x[1] = x[1] + v[1]*deltaT

    

def EulerCromer_3(t, x, v, a, deltaT, k, m, i):
    
    '''
    Auxiliary funtion of springCalEC(t, x, v, deltaT, k, m, n) for three masses and three springs
    '''
    
    a[0] = (-k[0]*x[0])/m[0] + (k[1]*(x[1]-x[0]))/m[0]
    a[1] = -(k[1]*(x[1]-x[0]))/m[1] + (k[2]*(x[2]-x[1]))/m[1]
    a[2] = -(k[2]*(x[2]-x[1]))/m[2] 
    
    v[0] = v[0] + a[0]*deltaT
    v[1] = v[1] + a[1]*deltaT
    v[2] = v[2] + a[2]*deltaT
    
    x[0] = x[0] + v[0]*deltaT
    x[1] = x[1] + v[1]*deltaT
    x[2] = x[2] + v[2]*deltaT


def springCalEC(t, x, v, deltaT, k, m, n):
    
    '''
    Goal: make calculate position, velocity and energy for each timestep using Euler-Cromer:  
    
    Variables: 
                t - array which will hold simulation timestep 
                x - array which will hold simulated positions for all masses (depending on the number of masses in use)
                v - array which will hold simulated velocities for all masses (depending on the number of masses in use)
                deltaT - time between iterations
                k - array containig k for each spring
                m - array containig m for each mass
                n - iteration number
    
    return:
                t - array with simulation timestep
                x - array with simulated positions for each iteration for all masses (depending on the number of masses in use)
                v - array with simulated velocities for each iteration for all masses (depending on the number of masses in use)
                E - array with simulated energies for each iteration
    '''
    
    a = np.zeros(len(x)) #len(x) is the number of masses and springs
    Emec = 0
    i = 0
    
    if len(x) == 1: #if there is only one spring and one mass in the simulation
        while i < n:
            t += deltaT
            EulerCromer_1(t, x, v, a, deltaT, k, m,  i)
            i += 1  
        Emec = (0.5*m[0]*v[0]**2) + (0.5*k[0]*x[0]**2)
        
    
    elif len(x) == 2: #if there are two springs and two masses in the simulation
        while i < n:
            t += deltaT
            EulerCromer_2(t, x, v, a, deltaT, k, m,  i)
            i += 1 
        Emec = (0.5*m[0]*v[0]**2)+ (0.5*m[1]*v[1]**2) + (0.5*k[1]*((x[1])-(x[0]) )**2) + (0.5*k[0]*(x[0])**2)
    
    else: #if there are three springs and three masses in the simulation
        while i < n: 
            t += deltaT
            EulerCromer_3(t, x, v, a, deltaT, k, m,  i)
            i += 1 
        Emec = (0.5*m[0]*v[0]**2)+ (0.5*m[1]*v[1]**2) + (0.5*k[1]*(x[1]-x[0])**2) + (0.5*k[0]*(x[0])**2) + (0.5*m[2]*v[2]**2) + (0.5*k[2]*(x[2]-x[1])**2)
    
    return t, x, v, Emec

#%% Beeman functions

def Beeman_1(t, x, v, a, deltaT, k, m,  i):
    
    '''
    Auxiliary funtion of springCalBM(t, x, v, deltaT, k, m, n ) for one mass and one springs
    ''' 

    a_previous = a[0]
    a[0] = (-k[0]*(x[0]))/m[0] #a(t)
    x[0] = x[0] + v[0]*deltaT +(2/3)*a[0]*(deltaT**2)-(1/6)*a_previous*deltaT*deltaT  
    
    a_next = (-k[0]*(x[0]))/m[0]
    v[0] = v[0] + (1/3)*a_next*deltaT + (5/6)*a[0]*deltaT -(1/6)*a_previous*deltaT

    
def Beeman_2(t, x, v, a, deltaT, k, m,  i): 
    
    '''
    Auxiliary funtion of springCalBM(t, x, v, deltaT, k, m, n ) for two masses and two springs
    ''' 
    
    a0_previous = a[0]
    a1_previous = a[1]
    
    a[0] = (-k[0]*x[0])/m[0] + (k[1]*(x[1]-x[0]))/m[0] #a0(t)
    a[1] = -(k[1]*(x[1]-x[0]))/m[1] #a1(t)
    
    x[0] = x[0] + v[0]*deltaT +(2/3)*a[0]*(deltaT**2)-(1/6)*a0_previous*deltaT*deltaT  
    x[1] = x[1] + v[1]*deltaT +(2/3)*a[1]*(deltaT**2)-(1/6)*a1_previous*deltaT*deltaT 
    
    a0_next = (-k[0]*x[0])/m[0] + (k[1]*(x[1]-x[0]))/m[0] 
    a1_next = -(k[1]*(x[1]-x[0]))/m[1]
    
    v[0] = v[0] + (1/3)*a0_next*deltaT + (5/6)*a[0]*deltaT -(1/6)*a0_previous*deltaT
    v[1] = v[1] + (1/3)*a1_next*deltaT + (5/6)*a[1]*deltaT -(1/6)*a1_previous*deltaT
    
    
def Beeman_3(t, x, v, a, deltaT, k, m,  i):  

    '''
    Auxiliary funtion of springCalBM(t, x, v, deltaT, k, m, n ) for three masses and three springs
    '''      
    
    a0_previous = a[0]
    a1_previous = a[1]
    a2_previous = a[2]
    
    a[0] = (-k[0]*x[0])/m[0] + (k[1]*(x[1]-x[0]))/m[0] #a0(t)
    a[1] = -(k[1]*(x[1]-x[0]))/m[1] + (k[2]*(x[2]-x[1]))/m[1] #a1(t)
    a[2] = -(k[2]*(x[2]-x[1]))/m[2] #a2(t)
    
    x[0] = x[0] + v[0]*deltaT +(2/3)*a[0]*(deltaT**2)-(1/6)*a0_previous*deltaT*deltaT  
    x[1] = x[1] + v[1]*deltaT +(2/3)*a[1]*(deltaT**2)-(1/6)*a1_previous*deltaT*deltaT 
    x[2] = x[2] + v[2]*deltaT +(2/3)*a[2]*(deltaT**2)-(1/6)*a2_previous*deltaT*deltaT 
    
    a0_next = (-k[0]*x[0])/m[0] + (k[1]*(x[1]-x[0]))/m[0] 
    a1_next = -(k[1]*(x[1]-x[0]))/m[1] + (k[2]*(x[2]-x[1]))/m[1]
    a2_next = -(k[2]*(x[2]-x[1]))/m[2]
    
    v[0] = v[0] + (1/3)*a0_next*deltaT + (5/6)*a[0]*deltaT -(1/6)*a0_previous*deltaT
    v[1] = v[1] + (1/3)*a1_next*deltaT + (5/6)*a[1]*deltaT -(1/6)*a1_previous*deltaT
    v[2] = v[2] + (1/3)*a2_next*deltaT + (5/6)*a[2]*deltaT -(1/6)*a2_previous*deltaT



def springCalBM(t, x, v, deltaT, k, m, n ):
    
    '''
    Goal: make calculate position, velocity and energy for each timestep using Beeman:  
    
    Variables: 
                t - array which will hold simulation timestep 
                x - array which will hold simulated positions for all masses (depending on the number of masses in use)
                v - array which will hold simulated velocities for all masses (depending on the number of masses in use)
                deltaT - time between iterations
                k - array containig k for each spring
                m - array containig m for each mass
                n - iteration number
    
    return:
                t - array with simulation timestep
                x - array with simulated positions for each iteration for all masses (depending on the number of masses in use)
                v - array with simulated velocities for each iteration for all masses (depending on the number of masses in use)
                E - array with simulated energies for each iteration
    '''
    
    a = np.zeros(len(x))
    Emec = 0
    #one euler:
    i = 0
    
    if len(x) == 1: #if there is only one spring and one mass in the simulation
        while i < n:
            t += deltaT
            if i == 0:#first values must be calculated via euler-cromer
                EulerCromer_1(t, x, v, a, deltaT, k, m, i) #a(t-dt); v(t); x(t)
                
            else:
                Beeman_1(t, x, v, a, deltaT, k, m, i)
            i += 1
        Emec = (0.5*m[0]*v[0]**2) + (0.5*k[0]*x[0]**2)
             
            
    elif len(x) == 2 : #if there are two springs and two masses in the simulation
        while i < n:
            t += deltaT
            if i == 0: #first values must be calculated via euler-cromer
                EulerCromer_2(t, x, v, a, deltaT, k, m, i) #a(t-dt); v(t); x(t)

            else: 
                Beeman_2(t, x, v, a, deltaT, k, m, i)
            i += 1    
        Emec = (0.5*m[0]*v[0]**2)+ (0.5*m[1]*v[1]**2) + (0.5*k[1]*((x[1])-(x[0]) )**2) + (0.5*k[0]*(x[0])**2)

            
    else: #if there are three springs and three masses in the simulation
        while i < n:
            t += deltaT
            if i == 0:#first values must be calculated via euler-cromer
                EulerCromer_3(t, x, v, a, deltaT, k, m, i) #a(t-dt); v(t); x(t)
              
            else: 
                Beeman_3(t, x, v, a, deltaT, k, m, i)
            i += 1
        Emec = (0.5*m[0]*v[0]**2)+ (0.5*m[1]*v[1]**2) + (0.5*k[1]*(x[1]-x[0])**2) + (0.5*k[0]*(x[0])**2) + (0.5*m[2]*v[2]**2) + (0.5*k[2]*(x[2]-x[1])**2)

        
    return t, x, v, Emec

#%% Main function
            
def springSim(inicial, deltaT, Tmax, graphT, mode):
    
    '''
    Goal: 
    
        
    Variables:
        
            inicial:  Array with information regarding the inicial displacements,
                      inicial velocities k values for each spring and m values for each mass 
            deltaT: time between iterations
            Tmax: total simulation time (in simulation units)
           graphT: time step to aqquire values
           mode: defines which simulation is running or if they are both running at onde
            
                # mode =0 for just euler-cromer
                # mode =1 for just beeman
                # mode !=1 and !=0 for both
            
    
    return:
            t - array with simulation timestep (euler-cromer)
            xData - array with simulated positions for all masses (euler-cromer)
            vData - array with simulated velocities for all masses (euler-cromer)
            EData - array with simulated energies (euler-cromer)
            tb -array with simulation timestep (beeman)
            xDatab- array with simulated positions for all masses (beeman)
            vDatab - array with simulated velocities for all masses (beeman) 
            EDatab - array with simulated energies (beeman)
        
    '''
    
    graphT = graphT if graphT > deltaT else deltaT
    size = int(Tmax/graphT) + 1
    
    t, xData, vData, EData = inicialization(size, inicial)
    tb, xDatab, vDatab, EDatab = inicialization(size, inicial)
    
    n = int(graphT / deltaT)
    
    k = inicial[2]
    m = inicial[3]
    
    im = 0
    index = 1
    while index < size:
        if mode == 0:
            t[index], xData[index], vData[index], EData[im] = springCalEC(t[im], xData[im], vData[im], deltaT, k, m,n)
        elif mode == 1:
            tb[index], xDatab[index], vDatab[index], EDatab[im] = springCalBM(tb[im], xDatab[im], vDatab[im], deltaT, k, m,n)
        else:
            t[index], xData[index], vData[index], EData[im] = springCalEC(t[im], xData[im], vData[im], deltaT, k, m,n)
            tb[index], xDatab[index], vDatab[index], EDatab[im] = springCalBM(tb[im], xDatab[im], vDatab[im], deltaT, k, m,n)
        im = index
        index += 1
        
    return t, xData, vData, EData , tb, xDatab, vDatab, EDatab

#%%Inicial Conditions

x_0 = [2.0,2.0, 2.0] #displacement in m
v_0 = [0.0,0.0,0.0] #inicial velocities of the masses in m/s
k = [10,10,10] #elasticity coeficient of the springs in N/m
m = [1,1,1] #mass of objects in kg

'''
To change the number of masses used in simulation from  1 to 3: 
    for one mass and one spring: make sure x_0 and v_0 only have one value each 
                    e.g ( x_0 = [2.0] and v_0 = [1.0])
                    
    for two masses and two springs: make sure x_0 and v_0 have two values each 
                    e.g ( x_0 = [2.0, 0.0] and v_0 = [1.0, -4.0])
                    
    for three masses and three springs: make sure x_0 and v_0 have three values each 
                    e.g ( currently displayed in the beginning of this cell)
                    
    k and m arrays can stay the same size
'''

inicial = [x_0, v_0, k, m ]

deltaT = 0.001
Tmax = 1000
graphT = 0.05


# mode =0 for just euler
# mode =1 for just beeman
# mode !=1 and !=0 for both

t, x, v, E, tb, xb, vb, Eb = springSim(inicial, deltaT, Tmax, graphT, 6)  


#%% aux fft plot funtions



def max_fft(ax,xmax,ymax, position1, position2):
    text= "x={:.4f}, y={:.4f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(position1,position2), **kw)

def annot_max(x, y, ax=None):
    x_biggest = x[np.argmax(y)]
    y_biggest = y.max()
    max_fft(ax,x_biggest,y_biggest, 0.94, 0.96)
    
def annot_sec_max(x, y,mode, ax=None):
    
        found1 = 0
        found2 = 0
             
        
        for w in range(1, len(y)-1):
            if (y[w- 1] < y[w] 
                and y[w + 1] < y[w] ):
        
                y_biggest = y[w]
                y_biggest_index = w
                found1 = 1
                print(w)
                break
        
        for i in range(w+1, len(y)-1):
            if (y[i- 1] < y[i] 
                and y[i + 1] < y[i]):
        
                y_second_biggest = y[i]
                y_second_biggest_index = i
                found2 = 2
                break
                

                
        labels = ['Euler-Cromer', 'Beeman FFT']
        if mode == 1:
            label_ = labels[0]
        else:
            label_ = labels[1]
        print('freq naturais ' + label_ )
        x_biggest = x[np.argmax(y)]
        print(x_biggest)
        
        if (found1 == 1):
            x_biggest = x[y_biggest_index]
            max_fft(ax,x_biggest,y_biggest, 0.94, 0.96)
        
        if (found2 == 2):
            x_second_biggest = x[y_second_biggest_index]
            max_fft(ax,x_second_biggest,y_second_biggest, 0.74, 0.76)
            
            print(x_second_biggest)

           
    
def annot_third_max(x, y, mode, ax=None):
    
        found1 = 0
        found2 = 0
        found3 = 0 
              
        for w in range(1, len(y)-1):
            if (y[w- 1] < y[w] 
                and y[w + 1] < y[w]):
        
                y_biggest = y[w]
                y_biggest_index = w
                found1 = 1
                
                break
            
       
        for i in range(w+1, len(y)-1):
            if (y[i- 1] < y[i] 
                and y[i + 1] < y[i]):
        
                y_second_biggest = y[i]
                y_second_biggest_index = i
                found2 = 2
            
                break
            
               
        for j in range(i+1, len(y)-1):
            if (y[j- 1] < y[j] 
                and y[j + 1] < y[j]):
                
                y_third_biggest = y[j]
                y_third_biggest_index = j
                found3 = 3
                break
            
           
        labels = ['Euler-Cromer', 'Beeman FFT']
        if mode == 1:
            label_ = labels[0]
        else:
            label_ = labels[1]
        print('freq naturais ' + label_ )
        x_biggest = x[np.argmax(y)]
        print(x_biggest)
        
        if (found1 == 1):
            x_biggest = x[y_biggest_index]
            max_fft(ax,x_biggest,y_biggest, 0.94, 0.96)
        
        if (found2 == 2):
            x_second_biggest = x[y_second_biggest_index]
            max_fft(ax,x_second_biggest,y_second_biggest, 0.74, 0.76)
            
            print(x_second_biggest)
        if (found3 == 3):
            x_third_biggest = x[y_third_biggest_index]
            max_fft(ax,x_third_biggest,y_third_biggest, 0.54, 0.56)
            print(x_third_biggest)
            
        
        
        

#%%fft plot function 

def fft_plot (x_0, x, mode, deltaT, Tmax, graphT, t ):
    
    labels = ['Euler-Cromer FFT', 'Beeman FFT']
    if mode == 1:
        label_ = labels[0]
    else:
        label_ = labels[1]
        
    N = len(x)
    T = t[-1]/len(t)
    amp_mass_0 = fft(x)
    freq = fftfreq(N, T)[:N//2]
    amp = 2.0/N * np.abs(amp_mass_0[0:N//2])
    figSR2, ax = plt.subplots(figsize = (8,8))
      
    if len(x_0) == 1:
        annot_max(freq,amp)
    elif len(x_0) ==2:
        annot_sec_max(freq,amp, mode)
    else:
        annot_third_max(freq, amp, mode)
    ax.plot(freq[0:100000000], amp[0:100000000], 'o-')
    
    ax.set_title( label = label_ + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
    ax.set_ylabel(r'Amplitude')
    ax.set_xlabel(r'Frequency (Hz)')
    

  


#%% Plots

x_plot =[]
x_t = x.transpose()
for i in range(0, len(x_0)):    
    x_plot.append(x_t[i])
    
xb_plot =[]
xb_t = xb.transpose()
for i in range(0, len(x_0)):    
    xb_plot.append(x_t[i])    

if len(x_plot) ==3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_plot[0], x_plot[1], x_plot[2])
    ax.set_title( label = 'Euler-Cromer' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('x2')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xb_plot[0], xb_plot[1], xb_plot[2])
    ax.set_title( label = 'Beeman' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('x2')

x_t = x.transpose()
xb_t = xb.transpose()

fig_position_euler_0, ax_position_euler_0 = plt.subplots(figsize = (8,8))
ax_position_euler_0.plot(t, x_t[0], 'o-')

ax_position_euler_0.set_title( label = 'Euler-Cromer - Position of mass 0 at' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
ax_position_euler_0.set_ylabel(r'Position of mass 0 in relation to its equilibrium position (m)')
ax_position_euler_0.set_xlabel(r'Time (s)')


fig_position_beeman_0, ax_position_beeman_0 = plt.subplots(figsize = (8,8))
ax_position_beeman_0.plot(tb, xb_t[0], 'o-')
ax_position_beeman_0.set_title( label = 'Beeman - Position of mass 0 at' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
ax_position_beeman_0.set_ylabel(r'Position of mass 0 in relation to its equilibrium position (m)')
ax_position_beeman_0.set_xlabel(r'Time (s)')

fig_position_dif_0, ax_position_dif_0 = plt.subplots(figsize = (8,8))
ax_position_dif_0.plot(tb, abs(x_t[0] - xb_t[0]), 'o-')
ax_position_dif_0.set_title( label = 'Diference (|e-b|) position of mass 0 at' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
ax_position_dif_0.set_ylabel(r'Diference of position of mass 0 in relation to its equilibrium position (m)')
ax_position_dif_0.set_xlabel(r'Time (s)')


E_t = E.transpose()
Eb_t = Eb.transpose()


fig_energy_euler, ax_energy_euler = plt.subplots(figsize = (8,8))
ax_energy_euler.plot(t[1:], E_t , 'o-')
ax_energy_euler.set_title( label = 'Euler-Cromer - Energy of system at' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
ax_energy_euler.set_ylabel(r'Energy (J)')
ax_energy_euler.set_xlabel(r'Time (s)')

fig_energy_beeman, ax_energy_beeman = plt.subplots(figsize = (8,8))
ax_energy_beeman.plot(tb[1:], Eb_t , 'o-')
ax_energy_beeman.set_title( label = 'Beeman - Energy of system at' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
ax_energy_beeman.set_ylabel(r'Energy (J)')
ax_energy_beeman.set_xlabel(r'Time (s)')


fig_energy, ax_energy = plt.subplots(figsize = (8,8))
e, = ax_energy.plot(t[1:], E_t , 'o-')
b, = ax_energy.plot(tb[1:], Eb_t , 'o-')
#ax_energy.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_energy.legend([e, b], ['Euler-Cromer', 'Beeman'], loc='upper right', shadow=True)
ax_energy.set_title( label = 'Comparison of energy of system at' + ' ' + '(deltaT = ' + str(deltaT) + ' ' + 'Tmax = ' + str(Tmax) + ' ' + 'graphT = ' + str(graphT) + ')' )
ax_energy.set_ylabel(r'Energy (J)')
ax_energy.set_xlabel(r'Time (s)')


#%% euler-cromer and beeman fft 
'''
Para alterar a massa que está a ser estuda no gráfico fft basta alterar os indices de x_t[] e xb_t[].
    0 para a massa zero
    1 para a massa um
    2 para a massa dois
'''
fft_plot (x_0, x_t[0], 1,deltaT, Tmax, graphT, t ) 
fft_plot (x_0, xb_t[0], 2,deltaT, Tmax, graphT, tb ) 









