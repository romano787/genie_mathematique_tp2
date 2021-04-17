"""
---------------------------
TP1 Génie Mathématiques
---------------------------
MIAUX Romain
PREMARAJAH Piratheban
2PF1
"""


import numpy as np
# from tp2 import (Cholesky, ResolCholesky)
import time
from matplotlib import pyplot as plt
import math


def ReductionGauss(Aaug):

    n, m = Aaug.shape

    for i in range(n-1):

        if Aaug[i, i] == 0:
            print("Un pivot est nul, prendre la méthode avec pivot partiel")
            return []

        else:
            for j in range(i+1, n):
                g = Aaug[j, i] / Aaug[i, i]
                Aaug[j, :] = Aaug[j, :] - g * Aaug[i, :]

    return Aaug


def ResolutionSystTriSup(Taug):
    """
    Donne la solution d'une matrice triangulaire supérieure

    Args:
        Taug (np.array(), matrice): matrice augmentée triangulaire supérieure

    Returns:
        X: solution du système
    """

    n, m = Taug.shape

    if Taug[n-1, n-1] == 0:  # On regarde si le dernier pivot est nul
        print("La matrice triangulaire supérieure n'est pas bonne")
        X = None        # On renvoit aucune solution car on ne peut pas résoudre ce genre
        # de système avec l'algorithme de Gauss simple

    else:
        X = np.zeros((n, 1))        # On crée notre matrice solution
        X[-1] = Taug[-1, -1] / Taug[n-1, n-1]   # On change le dernier terme

        # Puis on vient changer les valeurs des X[i] pour avoir au final notre solution du système
        for i in range(n-2, -1, -1):
            X[i] = Taug[i, -1]

            for j in range(i+1, n):
                X[i] = X[i] - Taug[i, j] * X[j]

            X[i] = X[i] / Taug[i, i]

    return X


def Gauss(A, B):
    """
    Donne la solution d'une équation AX = B, avec X la solution du système

    Args:
        A (np.array(), matrice): matrice carrée de taille n x n
        B (np.array(), matrice): matrice colonne de taille n x 1

    Returns:
        X: solution de l'équation AX = B
    """
    A = np.array(A, float)
    B = np.array(B, float)
    Aaug = np.column_stack((A, B))
    Taug = ReductionGauss(Aaug)
    X = ResolutionSystTriSup(Taug)

    return X


def DecompositionLU(A):
    """
    Décompose la matrice A en deux autres matrices L et U

    Args:
        A (np.array(), matrice): matrice carrée de taille n x n

    Returns:
        U, L: deux matrice de taille n x n, U étant une matrice triangulaire supérieure et L, une matrice triangulaire inférieure
    """
    A = np.array(A, float)
    n, m = A.shape
    # On crée une matrice identité pour L directement comme ça les un sont déjà placés sur
    # la diagonale
    L = np.identity(n)

    for i in range(n-1):

        if A[i, i] == 0:    # On vérifie que le pivot n'est pas nul
            print("Un pivot est nul, prendre la méthode avec pivot partiel")
            return []

        else:
            # On vient modifier la ligne j pour chaque colonne i
            for j in range(i+1, n):
                L[j, i] = A[j, i] / A[i, i]

                # Puis on vient modifier la matrice A avec ici la j-ième ligne pour la k-ième
                # colonne
                for k in range(i, n):
                    A[j, k] = A[j, k] - L[j, i] * A[i, k]

    U = A.copy()   # Le dernier A qu'on obtient est donc notre matrice U

    return U, L


def ResolutionLU(L, U, B):
    """
    Résoud le système avec les matrices L et U issues de la décomposition de A

    Args:
        L (np.array(), matrice): matrice triangulaire inférieure de taille n x n
        U (np.array(), matrice): matrice triangulaire supérieure de taille n x n
        B (np.array(), matrice): matrice colonne de taille n x 1

    Returns:
        X: solution du système
    """
    B = np.array(B, float)
    n, m = B.shape
    # On crée nos matrices solutions
    Y = np.ones((n, 1))
    X = np.ones((n, 1))

    # On résoud d'abord l'équation LY = B
    for i in range(n):

        sum_ = 0

        for j in range(i):
            sum_ = sum_ + L[i, j] * Y[j]

        Y[i] = B[i] - sum_

    # Puis, on vient résoudre UX = Y
    for i in range(n-1, -1, -1):

        sum_ = 0

        for j in range(i+1, n):
            sum_ = sum_ + U[i, j] * X[j]

        X[i] = (1 / U[i, i]) * (Y[i] - sum_)

    return X


def GaussChoixPivotPartiel(A, B):
    """
    Résolution de systèmes linéaires AX = B en effectuant des échanges de lignes sur A pour avoir le plus grand pivot possible sur chaque ligne à chaque fois

    Args:
        A (np.array(), matrice): matrice carrée de taille n x n
        B (np.array(), matrice)): matrice colonne de taille n x 1

    Returns:
        X: solution du système
    """
    A = np.array(A, float)
    B = np.array(B, float)
    n, m = A.shape

    for i in range(n):

        for j in range(i, n):

            # On regarde si le nombre de la ligne j sur la colonne i est plus grand que celui du pivot
            if abs(A[j, i]) > abs(A[i, i]):

                # Si oui, alors:
                temp = A[j, :].copy()   # On échange les lignes
                A[j, :] = A[i, :]
                A[i, :] = temp

                # On oublie pas de changer les lignes pour la matrcie B également
                temp1 = B[j].copy()
                B[j] = B[i]
                B[i] = temp1

        # On effectue les calculs normalement
        for j in range(i+1, n):

            g = A[j, i] / A[i, i]

            for k in range(i, n):
                A[j, k] = A[j, k] - g * A[i, k]

            B[j] = B[j] - g * B[i]

    # Puis, on vient faire notre matrice triangulaire supérieure augmentée
    Taug = np.column_stack((A, B))
    X = ResolutionSystTriSup(Taug)

    return X


def GaussChoixPivotTotal(A, B):
    """
    Résoud l'équation AX = B mais cette fois-ci en changeant les lignes ET les colonnes de A

    Args:
        A (np.array(), matrice): matrice carrée de taille n x n
        B (np.array(), matrice): matrice colonne de taille n x 1

    Returns:
        X: solution du système
    """

    A = np.array(A, float)
    B = np.array(B, float)
    n, m = A.shape

    # On crée une matrice de référence pour savoir où se situe nos variables (car les changements de colonnes changent l'ordre des variables)
    ref = np.arange(n)

    for i in range(n):

        for j in range(i, n):

            # On regarde si le nombre de la ligne i sur la colonne j est plus grand que
            # celui du pivot
            if abs(A[j, i]) > abs(A[i, i]):

                # Si oui, alors:
                temp = A[:, j].copy()   # On échange les colonnes
                A[:, j] = A[:, i]
                A[:, i] = temp

                # On n'oublie pas de changer aussi dans notre matrice de référence pour
                # suivre où sont les variables
                temp1 = ref[j].copy()
                ref[j] = ref[i]
                ref[i] = temp1

    # On fait maintenant les échanges de lignes
    X = GaussChoixPivotPartiel(A, B)

    # On remet nos variables dans le bonne ordre et pour ce faire, on utilise notre matrice de référence
    for i in range(n-1, -1, -1):  # On utilise un algorithme de tri

        for j in range(0, i):

            # Si nos nombres ne sont pas rangés dans l'ordre croissant, alors:
            if ref[j] > ref[j+1]:

                temp = ref[j]
                ref[j] = ref[j+1]
                ref[j+1] = temp

                # On change également la position de nos variables dans la matrice solution
                temp1 = X[j].copy()
                X[j] = X[j+1]
                X[j+1] = temp1

    return X


def PlotGauss():

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(50, 1050, 50):

        a = np.random.rand(i, i)
        b = np.random.rand(i, 1)

        start_time = time.time()

        x = Gauss(a, b)

        interval = time.time() - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(a, x) - b))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotLU(start=50, stop=1000, step=50):

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(start, stop + step, step):

        a = np.random.rand(i, i)
        b = np.random.rand(i, 1)

        start_time = time.time()

        U, L = DecompositionLU(a)
        x = ResolutionLU(L, U, b)

        interval = time.time() - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        # print(np.linalg.norm(np.dot(A, x) - b))
        list_error.append(np.linalg.norm(np.dot(a, x) - b))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotGaussPartiel():

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(50, 1050, 50):

        a = np.random.rand(i, i)
        b = np.random.rand(i, 1)

        start_time = time.time()

        x = GaussChoixPivotPartiel(a, b)

        interval = time.time() - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(a, x) - b))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotGaussTotal():

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(50, 550, 50):

        a = np.random.rand(i, i)
        b = np.random.rand(i, 1)

        start_time = time.time()

        x = GaussChoixPivotTotal(a, b)

        interval = time.time() - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(a, x) - b))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


def PlotCholesky():

    list_execution_speed_log = []
    list_execution_speed = []
    list_number = []
    list_number_log = []
    list_error = []

    for i in range(50, 550, 50):

        m = np.random.rand(i, i)
        A = np.dot(m, np.transpose(m))
        b = np.random.rand(i, 1)

        start_time = time.time()

        x = ResolCholesky(A, b)

        interval = time.time() - start_time

        print(f"Vitesse d'exécution : {interval}s")

        list_number_log.append(math.log(i))
        list_number.append(i)
        list_error.append(np.linalg.norm(np.dot(A, x) - b))
        list_execution_speed.append(interval)
        list_execution_speed_log.append(math.log(interval))

    return list_execution_speed_log, list_execution_speed, list_number, list_number_log, list_error


if __name__ == "__main__":

    # LES_log_G, LES_G, LN, LN_log, LE_G = PlotGauss()
    # LES_log_LU, LES_LU, LN, LN_log, LE_LU = PlotLU()
    LES_log_GP, LES_GP, LN, LN_log, LE_GP = PlotGaussPartiel()
    # LES_log_GT, LES_GT, LN, LN_log, LE_GT = PlotGaussTotal()
    # LES_log_C, LES_C, LN, LN_log, LE_C = PlotCholesky()

    # plt.plot(LN_log, LES_log_G, label="Pivot de Gauss simple")
    # plt.plot(LN_log, LES_log_LU, label="LU")
    plt.plot(LN_log, LES_log_GP, label="Pivot de Gauss partiel")
    # plt.plot(LN_log, LES_log_GT, label="Pivot de Gauss total")
    # plt.plot(LN_log, LES_log_C, label="Cholesky")
    plt.title(
        "Vitesse d'exécution de l'algorithme en fonction de la taille de la matrice")
    plt.xlabel("Log(n)")
    plt.ylabel("Log(t)")
    # plt.legend()
    plt.show()

    # -----------------------------------------------------------------------------------

    # plt.plot(LN, LES_G, label="Pivot de Gauss simple")
    # plt.plot(LN, LES_LU, label="LU")
    plt.plot(LN, LES_GP, label="Pivot de Gauss partiel")
    # plt.plot(LN, LES_GT, label="Pivot de Gauss total")
    # plt.plot(LN, LES_C, label="Cholesky")
    plt.title(
        "Vitesse d'exécution de l'algorithme en fonction de la taille de la matrice")
    plt.xlabel("n")

    # On indique ici qu'on veut les abscisses de 50 en 50
    plt.xticks(LN)
    # Rmq: On aurait pu utiliser toutes les autres listes contenant les valeurs de i

    # plt.legend()
    plt.ylabel("t")
    plt.show()

    # -----------------------------------------------------------------------------------

    # plt.plot(LN, LE_G, label="Pivot de Gauss simple")
    # plt.plot(LN, LE_LU, label="LU")
    plt.plot(LN, LE_GP, label="Pivot de Gauss partiel")
    # plt.plot(LN, LE_GT, label="Pivot de Gauss total")
    # plt.plot(LN, LE_C, label="Cholesky")
    plt.title("Erreur en fonction de la taille de la matrice")
    plt.xlabel("n")

    # On indique ici qu'on veut les abscisses de 50 en 50
    plt.xticks(LN)
    # Rmq: On aurait pu utiliser toutes les autres listes contenant les valeurs de i

    plt.ylabel("Erreur ||A*X - B||")
    plt.show()

    # np.linalg.solve(A, B)
