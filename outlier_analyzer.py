# File: outlier_analyzer.py
import pandas as pd
import numpy as np
from base_analyzer import BaseAnalyzer

class OutlierAnalyzer(BaseAnalyzer):
    def analyze_zscore(self, column=None, threshold=3):
        column = column or self.target_column
        print(f"\nPerforming Z-score analysis on {column}...")
        
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        outliers = self.df[z_scores > threshold].copy()
        outliers['method'] = 'Z-Score'
        outliers['score'] = z_scores[z_scores > threshold]
        outliers['analyzed_column'] = column
        return outliers

    def analyze_iqr(self, columns=None, factor=1.5):
        columns = columns or self.numerical_columns
        print(f"\nPerforming IQR analysis on {', '.join(columns)}...")
        
        outliers_all = pd.DataFrame()
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = self.df[
                (self.df[col] < (Q1 - factor * IQR)) | 
                (self.df[col] > (Q3 + factor * IQR))
            ].copy()
            
            if not outliers.empty:
                outliers['method'] = 'IQR'
                outliers['analyzed_column'] = col
                outliers['score'] = np.abs((outliers[col] - self.df[col].mean()) / self.df[col].std())
                outliers_all = pd.concat([outliers_all, outliers])
        
        return outliers_all