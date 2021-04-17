"""
---------------------------
TP2 Génie mathématique
---------------------------
MIAUX Romain
PREMARAJAH Piratheban
2PF1
"""


import numpy as np
import math
import time
from tp1 import *
from matplotlib import pyplot as plt


def Cholesky(A):
    """
    Décomposition de la matrice A en matrice L par la méthode de Cholesky

    Args:
        A (np.array()): matrice carrée de taille n x n

    Returns:
        L: matrice triangulaire inférieure carée de taille n x n 
    """

    A = np.array(A, float)
    n, m = A.shape
    L = np.zeros((n, n))

    for j in range(n):
        for i in range(j, n):

            # On regarde si on se trouve sur la diagonale de L et on fait :
            if i == j:
                L[i, j] = np.sqrt(A[i, j] - np.sum(L[i, :j]**2))

            # Sinon, pour toutes les autres valeurs supérieurs à j, on a :
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j]*L[j, :j])) / L[j, j]

    return L


def CholeskyAlternatif(A):
    """
    Décomposition de la matrice A par la méthode de Cholesky Alternatif, 
    c'est-à-dire que A = LDLt

    Args:
        A (np.array())): matrice carrée de taille n x n

    Returns:
        L: matrice triangulaire inférieure de taille n x n
        D: matrice diagonale de taille n x n
    """

    A = np.array(A, float)
    n, m = A.shape
    L = np.identity(n)
    D = np.zeros((n, n))

    # On utlise les formules données dans le partiel de 2019 pour calculer
    # les coefficients de L et D
    for k in range(n):
        sum_ = 0

        for j in range(k):
            sum_ = sum_ + ((L[k, j])**2) * D[j, j]

        D[k, k] = A[k, k] - sum_

        for i in range(k, n):
            sum_ = 0

            for j in range(k):
                sum_ = sum_ + L[i, j] * L[k, j] * D[j, j]

            L[i, k] = (1 / D[k, k]) * (A[i, k] - sum_)

    return L, D


def ResolCholeskyAlternatif(A, B):
    """
    Résoud l'équation AX = B avec la méthode de Cholesky Alternatif

    Args:
        A (np.array()): matrice carrée de taille n x n
        B (np.array()): matrice colonne de taille n x 1

    Returns:
        X: Solution du système 
    """

    B = np.array(B, float)
    L, D = CholeskyAlternatif(A)
    Lt = np.transpose(L)
    n, m = B.shape

    # On crée nos trois matrices qui vont nous servir à résoudre le système
    W = np.ones((n, 1))
    Y = np.ones((n, 1))
    X = np.ones((n, 1))

    # On calcule d'abord l'équation LW = B :
    for i in range(n):
        sum_ = 0

        for j in range(i):
            sum_ = sum_ + L[i, j] * W[j]

        W[i] = B[i] - sum_

    # Or W = DY donc on cherche Y ici :
    for i in range(n):
        Y[i] = W[i] / D[i, i]   # Comme D est diagonale, on divise seulement

    # Puis on résoud Y = Lt*X :
    for i in range(n-1, -1, -1):
        sum_ = 0

        for j in range(i+1, n):
            sum_ = sum_ + Lt[i, j] * X[j]

        X[i] = (1 / Lt[i, i]) * (Y[i] - sum_)

    return X


def ResolCholesky(A, B):
    """
    Résolution de l'équation AX = B par la méthode de Cholesky, 
    c'est-à-dire A = L*Lt

    Args:
        A (np.array()): matrice carrée de taille n x n
        B (np.array()): matrice colonne de taille n x 1

    Returns:
        x: Solution du système
    """

    L = Cholesky(A)
    U = np.transpose(L)

    n, m = np.shape(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # On calcule d'abord LY = B :
    for i in range(n):
        y[i] = (B[i] - np.sum(np.dot(L[i, :i], y[:i]))) / L[i, i]

    # Puis on calcule Y = Lt*X, ici U = Lt
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(np.dot(U[i, i+1:n], x[i+1:n]))) / U[i, i]

    # On met x en colonne de façon à ce que ce soit plus compréhensible
    x = np.reshape(x, (n, 1))

    return x


def NumpyCholesky(A, B):
    """
    Résolution de l'équation AX = B par la méthode de Cholesky, 
    c'est-à-dire A = L*Lt, en utilisant le module numpy ici

    Args:
        A (np.array()): matrice carrée de taille n x n
        B (np.array()): matrice colonne de taille n x 1

    Returns:
        x: Solution du système
    """

    # On utilise la fonction cholesky contenue dans la bibliothèque numpy
    L = np.linalg.cholesky(A)
    U = np.transpose(L)

    # Puis on fait comme avec l'alogorithme de Cholesky classique
    n, m = np.shape(L)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = (B[i] - np.sum(np.dot(L[i, :i], y[:i]))) / L[i, i]

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(np.dot(U[i, i+1:n], x[i+1:n]))) / U[i, i]

    x = np.reshape(x, (n, 1))

    return x


"""
Les fonctions ci-dessous vont avoir pour but de stocker 
les données des différentes méthodes pour pouvoir 
après les plot tous ensemble en bas
"""


def PlotCholesky(start=50, stop=1000, step=50):

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(start, stop + step, step):

        # Pour avoir une matrice symétrique définie positive,
        # on utilise une matrice aléatoire "m" et, pour avoir "A",
        # on fait A = m * mt
        m = np.random.rand(i, i)
        A = np.dot(m, np.transpose(m))
        B = np.random.rand(i, 1)

        start_time = time.perf_counter()

        x = ResolCholesky(A, B)

        stop_time = time.perf_counter()

        interval = stop_time - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(A, x) - B))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotNumpyCholesky(start=50, stop=1000, step=50):

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(start, stop + step, step):

        m = np.random.rand(i, i)
        A = np.dot(m, np.transpose(m))
        B = np.random.rand(i, 1)

        start_time = time.perf_counter()

        x = NumpyCholesky(A, B)

        stop_time = time.perf_counter()

        interval = stop_time - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(A, x) - B))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotCholeskyAlternatif(start=50, stop=1000, step=50):

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(start, stop + step, step):

        m = np.random.rand(i, i)
        A = np.dot(m, np.transpose(m))
        B = np.random.rand(i, 1)

        start_time = time.perf_counter()

        x = ResolCholeskyAlternatif(A, B)

        stop_time = time.perf_counter()

        interval = stop_time - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(A, x) - B))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotLinalgSolve(start=50, stop=1000, step=50):

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(start, stop + step, step):

        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        start_time = time.perf_counter()

        x = np.linalg.solve(A, B)

        stop_time = time.perf_counter()

        interval = stop_time - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(A, x) - B))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


if __name__ == "__main__":

    # On récupère les données des différentes méthodes
    LES_log_LU, LES_LU, LN, LN_log, LE_LU = PlotLU(stop=500)
    LES_log_C, LES_C, LN, LN_log, LE_C = PlotCholesky(stop=500)
    LES_log_CA, LES_CA, LN, LN_log, LE_CA = PlotCholeskyAlternatif(stop=500)
    LES_log_LS, LES_LS, LN, LN_log, LE_LS = PlotLinalgSolve(stop=500)
    LES_log_NC, LES_NC, LN, LN_log, LE_NC = PlotNumpyCholesky(stop=500)

    # Puis, on plot chaque méthode avec les données qui nous intéressent
    plt.plot(LN_log, LES_log_LU, label="LU")
    plt.plot(LN_log, LES_log_C, label="Cholesky")
    plt.plot(LN_log, LES_log_NC, label="Numpy Cholesky")
    plt.plot(LN_log, LES_log_CA, label="Cholesky alternatif")
    plt.plot(LN_log, LES_log_LS, label="numpy.linalg.solve")
    plt.title(
        "Vitesse d'exécution de l'algorithme en fonction de la taille de la matrice")
    plt.xlabel("Log(n)")
    plt.ylabel("Log(t)")
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------------------

    plt.plot(LN, LES_LU, label="LU")
    plt.plot(LN, LES_C, label="Cholesky")
    plt.plot(LN, LES_NC, label="Numpy Cholesky")
    plt.plot(LN, LES_CA, label="Cholesky alternatif")
    plt.plot(LN, LES_LS, label="numpy.linalg.solve")
    plt.title(
        "Vitesse d'exécution de l'algorithme en fonction de la taille de la matrice")
    plt.xlabel("n")

    # On indique ici qu'on veut les abscisses de 50 en 50
    plt.xticks(LN)
    plt.legend()
    plt.ylabel("t")
    plt.show()

    # -------------------------------------------------------------------------------------

    plt.plot(LN, LE_LU, label="LU")
    plt.plot(LN, LE_C, label="Cholesky")
    plt.plot(LN, LE_NC, label="Numpy Cholesky")
    plt.plot(LN, LE_CA, label="Cholesky alternatif")
    plt.plot(LN, LE_LS, label="numpy.linalg.solve")
    plt.title("Erreur en fonction de la taille de la matrice")
    plt.xlabel("n")

    # On indique ici qu'on veut les abscisses de 50 en 50
    plt.xticks(LN)
    plt.legend()
    plt.ylabel("Erreur ||A*X - B||")
    plt.show()
