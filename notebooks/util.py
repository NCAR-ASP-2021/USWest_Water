"""This is a general purpose module containing routines
(a) that are used in multiple notebooks; or 
(b) that are complicated and would thus otherwise clutter notebook design.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import socket

####################################################################################
### Import Packages ### run in tfp environment: 
####################################################################################


from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.keras.optimizer_v2.adam import Adam
tfd = tfp.distributions
import tensorflow.keras.backend as K
from tensorflow import math as tfm

import os
import glob
import sys
from scipy.stats import rankdata
import pandas as pd
import importlib
import copy
from netCDF4 import Dataset, num2date
from scipy.interpolate import interpn
from matplotlib.colors import Normalize 
from matplotlib import cm
import matplotlib as mpl
from math import erf


import matplotlib
#mapping
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import xarray as xr

####################################################################################
####################################################################################



def is_ncar_host():
    """Determine if host is an NCAR machine."""
    hostname = socket.getfqdn()
    
    return any([re.compile(ncar_host).search(hostname) 
                for ncar_host in ['cheyenne', 'casper', 'hobart']])


def build_nn_model(n_features, n_outputs, hidden_nodes, emb_size, optimizer='adam', lr=0.0001,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    activation='relu', reg=None):
    """
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        hidden_nodes: int or list of hidden nodes
        emb_size: Embedding size
        max_id: Max embedding ID
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
        activation: Activation function for hidden layer
    Returns:
        model: Keras model
    """
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    features_in = tf.keras.layers.Input(shape=(n_features,))
    x = features_in
    for h in hidden_nodes:
        x = tf.keras.layers.Dense(h, activation=activation, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dense(n_outputs, activation='linear', kernel_regularizer=reg)(x)
    model = tf.keras.models.Model(inputs=[features_in], outputs=x)
    return model