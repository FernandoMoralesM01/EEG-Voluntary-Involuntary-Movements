import numpy as np
import iberoSignalPro.preprocesa as ib
import matplotlib.pyplot as plt
import mne
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
import networkx as nx

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import contextlib
import os


def obtener_win(sig, binary_sig, siPlot=True):
        # Ensure binary_sig is binary
        binary_sig = np.array(binary_sig).flatten()
        binary_sig = (binary_sig >= 0.5).astype(int)  # Convert to binary (0 or 1)

        diff = np.diff(binary_sig)
        idx_actividad = np.where(diff == 1)[0]
        idx_rep = np.where(diff == -1)[0]

        print(binary_sig.shape)

        if binary_sig[0] == 1:
            idx_actividad = np.insert(idx_actividad, 0, 0)
        #if binary_sig[-1] == 1:
        #    idx_actividad = np.append(idx_actividad, len(binary_sig) -1)
        
        if binary_sig[0] == 0:
            idx_rep = np.insert(idx_rep, 0, 0)
        #if binary_sig[-1] == 0:
        #    idx_rep = np.append(idx_rep, len(binary_sig) - 1)
        
        #print(idx_actividad)
        #print(idx_rep)
        
        # Ensure idx_actividad and idx_rep have the same length by adding samples
        while len(idx_actividad) < len(idx_rep):
            idx_actividad = np.append(idx_actividad, idx_actividad[-1])
        while len(idx_rep) < len(idx_actividad):
            idx_rep = np.append(idx_rep, idx_rep[-1])
        
        if idx_rep[0] < idx_actividad[0]:
            ventanas_reposo = np.stack((idx_rep, idx_actividad)).T
            ventanas_actividad = np.stack((idx_actividad[:-1], idx_rep[1:])).T
        else:
            ventanas_reposo = np.stack((idx_rep[:-1], idx_actividad[1:])).T
            ventanas_actividad = np.stack((idx_actividad[:], idx_rep[:])).T

        #print(ventanas_reposo.shape)
        #print(ventanas_actividad.shape)

        if siPlot:
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 2, 1) 
            for ventana in ventanas_actividad:
                plt.plot(sig[ventana[0]: ventana[1]])
            plt.title('Ventanas de actividad')

            plt.subplot(1, 2, 2)  
            for ventana in ventanas_reposo:
                plt.plot(sig[ventana[0]: ventana[1]])
            plt.title('Ventanas de reposo')

            plt.show()

        return ventanas_actividad, ventanas_reposo


class Network:
    def __init__(self, data, bin, fs=10, ch_names=None, matriz_act=None, matriz_rep=None, densidad_act_prom=None, densidad_rep_prom = None, densidad_act_ch_in=None, densidad_act_ch_out=None, densidad_rep_ch_in=None, densidad_rep_ch_out=None, band = None):
        self.data = data
        self.bin = bin
        self.fs = fs
        self.ch_names = ch_names
        self.band = band
        
    def realizagranger(self, df, maxlag=6, pval=0.01, sel1="HRV", sel2="HRV"):
        try:
            # Verificar si hay suficientes datos
            if len(df) <= maxlag:
                
                print(f"{len(df)} Datos insuficientes?.")
                return 0
            
            # Realizar la prueba de causalidad de Granger
            gc_test_1 = grangercausalitytests(df[[sel1, sel2]], maxlag=maxlag, verbose=False)
            p_values = [gc_test_1[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            
            # Verificar si la media de los p-valores es menor que el umbral
            for val in p_values:
                if val < pval:
                    return 1
            return 0
            #return int(np.mean(p_values) < pval)
            
        except Exception as e:
            print(f"Error en ventana {e}")
            return 0
    
    def crea_matriz(self, df):
        matriz = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i != j:
                    matriz[i, j] = self.realizagranger(df, sel1=df.columns[i], sel2=df.columns[j])
        return matriz
    
    def densidad_red(self, actividades):
        densidades = [np.count_nonzero(actividad) / (actividad.shape[0] * actividad.shape[1]) for actividad in actividades]
        return np.mean(densidades)
    
    def densidad_channel(self, redes, canal, mode="input"):
        densidad_canal = []
        for red in redes:
            if mode == "input":
                densidad_canal.append(np.count_nonzero(red[canal, :]) / red.shape[0])
            elif mode == "output":
                densidad_canal.append(np.count_nonzero(red[:, canal]) / red.shape[1])
        return np.mean(densidad_canal)
    
    def get_ntwks(self, df, bin):
        df = df.fillna(0)
        ventanas_actividad, ventanas_reposo = obtener_win(bin, bin, siPlot=False)

        for i, ventanas in enumerate(ventanas_reposo):
            if ventanas[1] < ventanas[0]:
                if i == len(ventanas_reposo) - 1:
                    ventanas_reposo = ventanas_reposo[:i]
        
        for i, ventanas in enumerate(ventanas_actividad):
            if ventanas[1] < ventanas[0]:
                if i == len(ventanas_actividad) - 1:
                    ventanas_actividad = ventanas_actividad[:i] 
                                 
        window_len = 15

        diff_actividad = np.diff(ventanas_actividad, axis=1)
        diff_reposo = np.diff(ventanas_reposo, axis=1)

        len_window = window_len * self.fs
        #len_window = int(diff_reposo[diff_reposo > self.fs * window_len].mean())
        
        #max_ventanas = min(len(diff_actividad), len(diff_reposo[diff_reposo > self.fs * window_len]))

        #print(max_ventanas)
        #ventanas_actividad = ventanas_actividad[:max_ventanas]
        #ventanas_reposo = ventanas_reposo[:max_ventanas]

        actividades = []
        for i, ventana in enumerate(ventanas_actividad):
            len_window_temp = ventana[1] - ventana[0]
            if len_window_temp >= len_window:
                fill = (len_window_temp - len_window) // 2
                df_actividad = df.iloc[ventana[0] + fill: ventana[1] - fill, :]
                print("****************************")
                print(ventana[0] + fill, ventana[1] - fill)

                print(df_actividad.shape)
                print("****************************")
                
                matriz = self.crea_matriz(df_actividad)

                actividades.append(matriz)
        actividades = np.array(actividades)
        
        reposos = []
        for i, ventana in enumerate(ventanas_reposo):
            len_window_temp = ventana[1] - ventana[0]
            if len_window_temp >= len_window:
                fill = (len_window_temp - len_window) // 2
                df_reposo = df.iloc[ventana[0] + fill: ventana[1] - fill, :]

                print("****************************")
                print(ventana[0] + fill, ventana[1] - fill)
                
                print(df_reposo.shape)
                print("****************************")
                
                matriz = self.crea_matriz(df_reposo)
                reposos.append(matriz)
                
        reposos = np.array(reposos)
        

        self.matriz_act = np.sum(actividades, axis=0)
        self.matriz_rep = np.sum(reposos, axis = 0)

        self.densidad_act_prom = self.densidad_red(actividades)
        self.densidad_rep_prom = self.densidad_red(reposos)

        self.densidad_act_ch_in = [self.densidad_channel(actividades, i, mode="input") for i in range(actividades[0].shape[0])]
        self.densidad_act_ch_out = [self.densidad_channel(actividades, i, mode="output") for i in range(actividades[0].shape[0])]
        self.densidad_rep_ch_in = [self.densidad_channel(reposos, i, mode="input") for i in range(reposos[0].shape[0])]
        self.densidad_rep_ch_out = [self.densidad_channel(reposos, i, mode="output") for i in range(reposos[0].shape[0])]

        self.array_mat_act = actividades
        self.array_mat_rep = reposos
        return actividades, reposos
    
class Registro:
    def __init__(self, fs):
        """
        Inicializa la clase Registro.

        :param fs: Frecuencia de muestreo.
        """
        self.mu_networks = {}
        self.beta_networks = {}
        self.gamma_networks = {}
        self.fs = fs

    def add_network(self, name, network, tipo):
        """
        Agrega una red a la colección de redes.

        :param name: Nombre de la red.
        :param network: Instancia de la clase Network.
        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        """
        if tipo == 'mu':
            self.mu_networks[name] = network
        elif tipo == 'beta':
            self.beta_networks[name] = network
        elif tipo == 'gamma':
            self.gamma_networks[name] = network
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

    def get_network(self, name, tipo):
        """
        Obtiene una red de la colección de redes.

        :param name: Nombre de la red.
        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        :return: Instancia de la clase Network.
        """
        if tipo == 'mu':
            return self.mu_networks.get(name)
        elif tipo == 'beta':
            return self.beta_networks.get(name)
        elif tipo == 'gamma':
            return self.gamma_networks.get(name)
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

    def remove_network(self, name, tipo):
        """
        Elimina una red de la colección de redes.

        :param name: Nombre de la red.
        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        """
        if tipo == 'mu':
            if name in self.mu_networks:
                del self.mu_networks[name]
        elif tipo == 'beta':
            if name in self.beta_networks:
                del self.beta_networks[name]
        elif tipo == 'gamma':
            if name in self.gamma_networks:
                del self.gamma_networks[name]
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

    def list_networks(self, tipo):
        """
        Lista todas las redes en la colección de redes.

        :param tipo: Tipo de red ('mu', 'beta', 'gamma').
        :return: Lista de nombres de las redes.
        """
        if tipo == 'mu':
            return list(self.mu_networks.keys())
        elif tipo == 'beta':
            return list(self.beta_networks.keys())
        elif tipo == 'gamma':
            return list(self.gamma_networks.keys())
        else:
            raise ValueError("Tipo de red no reconocido. Use 'mu', 'beta' o 'gamma'.")

from collections import deque


def bfs_shortest_paths(graph, start):
    """Breadth-First Search to find all shortest paths from start node in a directed graph."""
    num_nodes = graph.shape[0]
    dist = [-1] * num_nodes
    dist[start] = 0
    paths = [[] for _ in range(num_nodes)]
    paths[start] = [[start]]
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        for neighbor in range(num_nodes):
            if graph[current, neighbor] > 0:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
                if dist[neighbor] == dist[current] + 1:
                    for path in paths[current]:
                        paths[neighbor].append(path + [neighbor])
    
    return paths

def betweenness_centrality(graph):
    num_nodes = graph.shape[0]
    betweenness = np.zeros(num_nodes)
    
    for s in range(num_nodes):
        paths = bfs_shortest_paths(graph, s)
        for t in range(num_nodes):
            if s != t:
                num_paths = len(paths[t])
                if num_paths > 0:
                    node_counts = np.zeros(num_nodes)
                    for path in paths[t]:
                        for node in path[1:-1]:
                            node_counts[node] += 1
                    betweenness += node_counts / num_paths
    
    return betweenness

def get_deg(test_mat):
    temp_in_deg = np.zeros((test_mat.shape[0]))
    temp_out_deg = np.zeros((test_mat.shape[0]))
    
    for i in range(test_mat.shape[0]):
        temp_in_deg[i] = np.sum(test_mat[:, i]) / (test_mat.shape[0] - 1)
        temp_out_deg[i] = np.sum(test_mat[i, :]) / (test_mat.shape[0] - 1)
    return temp_in_deg, temp_out_deg