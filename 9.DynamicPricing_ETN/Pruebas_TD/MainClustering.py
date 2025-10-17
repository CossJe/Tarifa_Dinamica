# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:53:08 2025

@author: Jesus Coss
"""
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import Tools4Cluster as TC
import Tools4ClasSupervisada as TCS

    
# Hace el clustering de todos los datos que salen del extractor
TC.ClusteringData(False)
# Hace el entrenamiento del algoritmo suoervisado 
TCS.ClusteringSupervisado
