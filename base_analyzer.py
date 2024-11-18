# File: base_analyzer.py
from datetime import datetime
import os
import pandas as pd
import numpy as np

class BaseAnalyzer:
    def __init__(self, df, date_column, id_column, numerical_columns, target_column, results_dir):
        self.df = df
        self.date_column = date_column
        self.id_column = id_column
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.results_dir = results_dir