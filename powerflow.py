
'''Simón Cometto 11/10/2019 Río Cuarto'''

# coding=utf-8
import numpy as np
import scipy.sparse as sparse
from math import cos, sin

class powerflow:
    '''

    '''
    def __init__(self, filename=''):
        with open(filename) as cdf:
            # Leo el archivo hasta llegar a la sección de BUS DATA
            words = ['', ]
            while words[0] != 'BUS':
                line = cdf.readline()
                words = line.split(' ')
                words = [item for item in words if item]  # Elimino los elementos vacios

            # Leo la cantidad de nodos en la 4ta columna
            self.n = int(words[3])
            n = self.n
            # Creo la Ybus (matriz de admitancia) nula de n x n numeros complejos
            self.Ybus = np.zeros((n, n), dtype=np.complex128)

            # Creo los vectores con las variables en cada nodo:
            self.load = np.zeros(n, dtype=np.complex128)  # P + jQ
            self.generation = np.zeros(n, dtype=np.complex128)  # P + jQ
            self.voltage = np.zeros(n, dtype=np.float)  # V(por unidad), angulo en grados
            self.angle = np.zeros(n, dtype=np.float)  # Angulo en grados

            self.PV_buses = np.array((0,2), dtype=int)  #Vector que contiene el índice del nodo PV, y la tensión del nodo

            self.Q_inj = np.zeros(n, dtype=np.float64)
            self.P_inj = np.zeros(n, dtype=np.float64)

            #Inicializo el valor del swing bus, pero en un nodo que no existe
            self.swing_bus = n+1

            # Leo las siguientes n lineas con la info de cada nodo
            for i in range(n):
                line = cdf.readline()
                words = line.split(' ')
                words = [item for item in words if item]  # Elimino los elementos vacios

                self.voltage[i] = float(words[7])
                self.angle[i] = np.deg2rad(float(words[8]))
                self.load[i] = complex(float(words[9]), float(words[10]))
                self.generation[i] = complex(float(words[11]), float(words[12]))

                self.Q_inj[i] = self.generation[i].imag - self.load[i].imag
                self.P_inj[i] = self.generation[i].real - self.load[i].real

                # Asigno el swing_bus
                if (int(words[6]) == 3):
                    self.swing_bus = i
                    self.swing_bus_angle = self.angle[i]
                    self.swing_bus_voltage = self.voltage[i]
                    #Como en los PV buses no se conoce ni P ni Q, se asignan valores nulos
                    self.P_inj[i] = 0
                    self.Q_inj[i] = 0

                # PV buses
                if (int(words[6]) == 2):
                    self.PV_buses = np.vstack((self.PV_buses, [i,float(words[14])])) #El índice y la tensión del bus
                    self.Q_inj[i] = 0 #Como en los PV buses se desconoce Q, se asigno un valor nulo

            # Leo el archivo hasta llegar a la sección de BRANCH DATA
            while words[0] != 'BRANCH':
                line = cdf.readline()
                words = line.split(' ')
                words = [item for item in words if item]  # Elimino los elementos vacios

            # Leo las lineas de la sección Branch
            while True:  # Salgo con un break en el próximo if
                line = cdf.readline()
                words = line.split(' ')
                words = [item for item in words if item]  # Elimino los elementos vacios

                # Si llego al fin de la sección indicado por un -999\n salgo del bucle
                if words[0] == '-999\n':
                    break

                i = int(words[0]) - 1
                j = int(words[1]) - 1  # La impedancia entre el nodo i y el nodo j
                self.Ybus[i, j] = self.Ybus[j, i] = -1 / complex(float(words[6]), float(
                    words[7]))  # Asigno la impendancia R + jX
                self.Ybus[i, i] = self.Ybus[j, j] = complex(0, float(
                    words[8]))  # En la diagonal sumo Charging B ''la impedancia paralelo del equivalente pi''

            # Recorro la matriz de admitacnia para asignarle a la diagonal la suma de las filas
            for i in range(0, n):
                for j in range(0, n):
                    if j != i:
                        self.Ybus[i, i] += -self.Ybus[i, j]

            self.init_v_theta()
            #np.savetxt('Ybus.txt', self.Ybus, fmt='%+9.4f', delimiter='  ')
            return

    def init_v_theta(self, init_voltage=1, init_angle=0):
        self.v = np.empty(self.n, dtype=np.float64)
        self.theta = np.empty(self.n, dtype=np.float64)

        for i in range(self.n):
            self.v[i] = init_voltage
            self.theta[i] = init_angle

            if np.any(self.PV_buses[:,0]==i):
                l = np.argwhere(self.PV_buses[:,0]==i)
                self.v[i] = self.PV_buses[l[0],1]

            if i == self.swing_bus:
                self.theta[i] = self.swing_bus_angle
                self.v[i] = self.swing_bus_voltage

    def reducir(self, x):
        '''Elimina las filas (y columas si es una matrix) que corresponden a Q del jacobiano y a V'''
        # Reducir un vector
        if x.ndim == 1:
            PV_buses_Q = self.PV_buses[:, 0] + self.n - 1
            filas_a_eliminar = np.append([self.swing_bus], [self.swing_bus + self.n - 1], )
            filas_a_eliminar = np.append(filas_a_eliminar, np.int32(PV_buses_Q))
            return np.delete(x, filas_a_eliminar, 0)

        # Reducir una matriz
        else:
            PV_buses_Q = self.PV_buses[:, 0] + self.n - 1
            filas_a_eliminar = np.append([self.swing_bus], [self.swing_bus+self.n-1], )
            filas_a_eliminar = np.append(filas_a_eliminar, np.int32(PV_buses_Q))

            columnas_a_eliminar = filas_a_eliminar
            x = np.delete(x, filas_a_eliminar, 0)
            return np.delete(x, columnas_a_eliminar, 1)

    def J(self):
        '''Computa el jacobiano para un valor de tensión y ángulo dado
        :parameter x: un vactor de 2*(n-1) donde n es la cantidad de nodos del sistema
        :returns jacobiano: una matriz de 2(n-1) x 2(n-1)
        '''

        #Cuatro matrices cuadradadas que despues se unen para formar el jacobiano
        J11 = np.zeros((self.n, self.n), dtype=np.float64)
        J12 = np.zeros((self.n, self.n), dtype=np.float64)
        J21 = np.zeros((self.n, self.n), dtype=np.float64)
        J22 = np.zeros((self.n, self.n), dtype=np.float64)

        for i in range(self.n):
            for j in range(self.n):
                # Saltear el swing_bus
                if (i == self.swing_bus or j == self.swing_bus):
                    continue

                # Elementos que no son de la diagonal
                # ---------------------------------------------------------------------------------------------
                if (i != j):
                    v_i = self.v[i]
                    v_j = self.v[j]
                    theta_i = self.theta[i]
                    theta_j = self.theta[j]
                    delta_theta = theta_i - theta_j
                    G_ij = self.Ybus[i,j].real
                    B_ij = self.Ybus[i,j].imag

                    cos_theta = cos(delta_theta)
                    sin_theta = sin(delta_theta)

                    a = v_i * v_j
                    b = a * G_ij
                    c = a * B_ij
                    # dP/dtheta
                    J11[i, j] = b * sin_theta - c * cos_theta
                    # dQ/dtheta
                    J21[i, j] = -b * cos_theta + c * sin_theta

                    d = v_i * G_ij
                    e = v_i * B_ij
                    # dP/dV
                    J12[i, j] = d * cos(delta_theta) + e * sin(delta_theta)
                    # dQ/dV
                    J22[i, j] = d * sin(delta_theta) - e * cos(delta_theta)

                # Elementos de la diagonal
                # ---------------------------------------------------------------------------------------------
                else:
                    v_i = self.v[i]
                    G_ii = self.Ybus[i,i].real
                    B_ii = self.Ybus[i,i].imag

                    P_i = self.last_P[i]
                    Q_i = self.last_Q[i]

                    # dP/dtheta
                    J11[i, j] = - Q_i - B_ii * (v_i ** 2)
                    # dP/dV
                    J21[i, j] = P_i / v_i + G_ii * v_i
                    # dQ/dtheta
                    J21[i, j] = P_i - G_ii * (v_i ** 2)
                    # dQ/dV
                    J22[i, j] = Q_i / v_i - B_ii * v_i

                # --------------------------------------------------------------------------------
        np.savetxt('jacobiano11.txt', J12, fmt='%+7.2f', delimiter='   ')

        J1 = np.hstack([J11, J12])
        J2 = np.hstack([J21, J22])
        J = np.vstack([J1, J2])

        return J

    def f(self):
        ''' Computa deltaP y deltaQ para un valor de tensión y ángulo dado
        :parameter x un vactor de 2*(n-1) donde n es la cantidad de nodos del sistema
        :returns delta_PQ: una vector de 2(n-1)'''

        P = np.zeros(self.n, dtype=np.float)
        Q = np.zeros(self.n, dtype=np.float)

        for i in range(self.n):
            for j in range(self.n):

                if (i == self.swing_bus): # Saltear el swing_bus
                    continue

                is_PV_bus = False #Variable para indicar si es un PV bus o no.
                if (np.any(self.PV_buses[:,0]==i)):
                    is_PV_bus = True

                #Se leen todas las variables necesarias
                B_ij = self.Ybus[i,j].imag
                G_ij = self.Ybus[i,j].real
                theta_i = self.theta[i]
                theta_j = self.theta[j]
                delta_theta = theta_i - theta_j
                v_i = self.v[i]
                v_j = self.v[j]
                a = v_i * v_j * G_ij
                b = v_i * v_j * B_ij

                #Se calcula y asignan los valores
                P[i] += a * cos(delta_theta) + b * sin(delta_theta)
                if not is_PV_bus:  #Si no es un PV_bus entonces se calcula Q
                    Q[i] += a * sin(delta_theta) - b * cos(delta_theta)

        #Guardo estas dos copias para luego usarlas en el cálculo de las diagonales del Jacobiano
        self.last_P = P * 100
        self.last_Q = Q * 100

        return (self.P_inj - P*100, self.Q_inj - Q*100)

    def solve_newton(self, init_v=1, init_angle=0):
        self.init_v_theta(init_v, init_angle)

        P, Q = self.f()
        f = np.append(P,Q)
        f_reducido = self.reducir(f)
        J_reducido = self.reducir(self.J())
        delta_x = np.linalg.solve(J_reducido, f_reducido)

        x = np.append(self.theta, self.v)
        x = delta_x - self.reducir(x)
        return x

    def disp_matrix(self, mat):
        '''Representa la topología de la matriz mediante un mapa de bits'''
        #Se quiero mostrar un vector, apilo uno arriba de otro para representarlo como una matriz
        if mat.ndim == 1:
            mat = np.vstack((mat, mat))
        
        import matplotlib.pyplot as plt
        if sparse.issparse(mat):
            mat = mat.todense()

        mat_plot = mat != 0.0
        plt.matshow(mat_plot)
        plt.show()


#---------------------------------------------------------------------------------------------

if __name__ == '__main__':

    ieee14bus = powerflow('IEEE14cdf.txt')

    x = ieee14bus.solve_newton()
    print(x)
    print(np.around(ieee14bus.last_P, 2))
    print(ieee14bus.P_inj)

    J = ieee14bus.J()
    #ieee14bus.disp_matrix(J)
    ieee14bus.disp_matrix(ieee14bus.reducir(J))
    #ieee14bus.disp_matrix(ieee14bus.Ybus)

    Yinv = np.linalg.inv(ieee14bus.Ybus)
    ieee14bus.disp_matrix(Yinv*10000000) #???