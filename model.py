#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:57:32 2022

@author: tobiastschuemperlin
"""

import pandas as pd 
import numpy as np

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import api_key, api_secret


client = Client(api_key, api_secret)