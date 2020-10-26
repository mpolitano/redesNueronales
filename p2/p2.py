import matplotlib.pyplot as plt
import numpy as np
import time
import pylab 

#https://mat.caminos.upm.es/wiki/Modelo_Depredador-Presa_de_Lotka-Volterra_(grupo_16)

class IF_Neuron:
    
    def __init__(self, Vrest, Vth, tau, R, i):
        self.Vrest = Vrest #E_l
        self.Vth = Vth
        self.tau = tau
        self.R = R
        self.current = i
    
    def V(self, V):
        return (self.Vrest - V + self.R * self.I())/self.tau     
    def V1(self, V):
        return (self.Vrest - V + self.R * self.I())/self.tau  
        R * I[i] *(1 - np.exp((-(t-t0)/t_m)))    
    
    def I(self):
        return self.current
    
    def coefsK(self, y, h):
        # Calcula los coeficientes
        k1 = self.V(y)
        k2 = self.V(y+.5*k1*h)
        k3 = self.V(y+.5*k2*h)
        k4 = self.V(y+k3*h)
        return k1, k2, k3, k4    
    
    def solve(self, y0, h, t_start, t_end,fire,rmd):
        t = t_start
        V = [y0]
        i = 0
        graph = []
        count = 0
        time = [] #tiempo de disparo
        while(t <= t_end):
        #while(i<2000):
        #RUNGE KUTTA
            if(rmd):
                self.current = np.random.randint(6)
            if(V[i] > self.Vth and fire):
                temp_y0 = self.Vrest
                count=count+1 #Numero de disparo o frecuencia
                time.append(t)
            else:
                k1, k2, k3, k4 = self.coefsK(V[i], h)
                temp_y0 = V[i] + h/6*(k1 + 2*k2 + 2*k3 + k4) #Rungekutta
                
            ## update values
            V.append(temp_y0)

            t += h
            i += 1
            graph.append([t, temp_y0])
        return graph,count

    def freq(self):
        I = np.linspace(0,6)
        count = []
        if_neuron = IF_Neuron(-65, -50, 10, 10, 2)

        # f = 1 / (self.t_ref - self.tau * np.log(1 - self.Vth/ (I * self.R)))
        for i in range(len(I)):
            if_neuron.setI(I[i])
            f,c =if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=True,rmd=False)
            # if (c != 0):
            firing_rate = c/200
            count.append(firing_rate)
            # else:
                # count.append(0)

        return I,count
    def getI(self):
        return self.current
    
    def setI(self, i):
        self.current = i
 

if_neuron = IF_Neuron(-65, -50, 10, 10, 2)

#Plot sin disparo 
# volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=False,rmd=False)
# volt = np.array(volt)
# f = plt.figure(figsize=(7, 3))
# plt.plot(volt[:,0], volt[:,1])
# plt.xlabel("Tiempo [ms]", fontsize = 7)
# plt.ylabel("Voltaje [mV]", fontsize = 7)
# plt.title("Respuesta ante una corriente de 2 nA en el tiempo 0-200. Sin Disparo", fontsize = 10)
# plt.savefig("sinDisparo")

# Plot con y sin disparo. Corriente es 2
f = plt.figure(figsize=(7, 3))
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=False,rmd=False)
volt = np.array(volt)
plt.plot(volt[:,0], volt[:,1])
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=True,rmd=False)
volt = np.array(volt)
plt.plot(volt[:,0], volt[:,1])
plt.xlabel("Tiempo [ms]", fontsize = 7)
plt.ylabel("Voltaje", fontsize = 7)
plt.title("Respuesta ante una corriente de 2 nA en el tiempo 0-200. Sin/Con Disparo", fontsize = 9)
plt.savefig("conSinDisparo")

#Plot Frecuenvia vs corriente
f = plt.figure(figsize=(5, 3))
I,c = if_neuron.freq()
plt.plot(I,c,'-')
# plt.plot(I[find( I == 6.0 )],c[2],'o')
plt.xlabel('Corriente externa', fontsize=7)
plt.ylabel('Frecuencia de disparo', fontsize=7)
plt.title("Frecuencia de disparo vs Corriente entrante", fontsize = 10)
plt.savefig("frecvscorri")


#Plot aleatorio. Con disparo
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=True,rmd=True)
volt = np.array(volt)
f = plt.figure(figsize=(7, 3))
plt.plot(volt[:,0], volt[:,1])
plt.xlabel("Tiempo [ms]", fontsize = 7)
plt.ylabel("Voltaje [mV]", fontsize = 7)
plt.title("Respuesta ante una corriente aleatoria en el tiempo 0-200.", fontsize = 10)
plt.savefig("aleatorio")

#Plot aleatorio. Con disparo y sin disparo
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=True,rmd=True)
volt = np.array(volt)
f = plt.figure(figsize=(7, 3))
plt.plot(volt[:,0], volt[:,1])
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=False,rmd=True)
volt = np.array(volt)
plt.plot(volt[:,0], volt[:,1])
plt.xlabel("Tiempo [ms]", fontsize = 7)
plt.ylabel("Voltaje [mV]", fontsize = 7)
plt.title("Respuesta ante una corriente aleatoria en el tiempo 0-200.", fontsize = 10)
plt.savefig("aleatorioConSin")

# if_neuron.setI(5e-9)
# volt = if_neuron.solve(y0=-65, h=0.1e-3, t_start=0, t_end=200e-3)
# volt = np.array(volt)
# f = plt.figure(figsize=(15, 5))
# plt.plot(volt[:,0], volt[:,1])
# plt.xlabel("Tiempo [ms]", fontsize = 15)
# plt.ylabel("Voltaje [mV]", fontsize = 15)
# plt.title("Respuesta Neurona Integrate and Fire ante I = 5 nA", fontsize = 20)


#Plot varias corrientes
f = pylab.figure(figsize=(7, 3))
if_neuron.setI(1.5)
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=False,rmd=False)
volt = np.array(volt)
pylab.plot(volt[:,0], volt[:,1], label='1.5')
if_neuron.setI(1.6)
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=True,rmd=False)
volt = np.array(volt)
pylab.plot(volt[:,0], volt[:,1], label='1.6')
if_neuron.setI(4)
volt,c = if_neuron.solve(y0=-65, h=0.05, t_start=0, t_end=200,fire=True,rmd=False)
volt = np.array(volt)
pylab.plot(volt[:,0], volt[:,1], label='4')
pylab.legend(loc='upper left')
pylab.xlabel("Tiempo [ms]", fontsize = 7)
pylab.ylabel("Voltaje", fontsize = 7)
pylab.title("Respuesta ante distintas corriente (1.5,1.6,3 mV).", fontsize = 9)
pylab.savefig("graficos")
pylab.show()