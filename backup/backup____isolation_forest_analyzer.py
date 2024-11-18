# File: isolation_forest_analyzer.py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from base_analyzer import BaseAnalyzer
import pandas as pd
import numpy as np

class IsolationForestAnalyzer(BaseAnalyzer):
    def analyze(self, single_column=None, multi_columns=None, contamination=0.01):
        single_column = single_column or self.target_column
        multi_columns = multi_columns or self.numerical_columns
        
        print("\nPerforming Isolation Forest analysis...")
        return {
            'single_feature': self._analyze_single_feature(single_column, contamination),
            'multi_feature': self._analyze_multi_feature(multi_columns, contamination)
        }
    
    def _analyze_single_feature(self, column, contamination):
        print(f"Running single-feature analysis on {column}...")
        
        values = self.df[[column]].values
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        scores = iso_forest.fit_predict(values)
        anomaly_scores = iso_forest.score_samples(values)
        
        outliers = self.df[scores == -1].copy()
        outliers['anomaly_score'] = anomaly_scores[scores == -1]
        outliers['analysis_type'] = 'Single-feature'
        outliers['analyzed_column'] = column
        
        return outliers

    def _analyze_multi_feature(self, columns, contamination):
        print(f"Running multi-feature analysis...")
        
        X_multi = self.df[columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_multi)
        
        iso_forest_multi = IsolationForest(contamination=contamination, random_state=42)
        multi_scores = iso_forest_multi.fit_predict(X_scaled)
        multi_anomaly_scores = iso_forest_multi.score_samples(X_scaled)
        
        outliers = self.df[multi_scores == -1].copy()
        outliers['anomaly_score'] = multi_anomaly_scores[multi_scores == -1]
        outliers['analysis_type'] = 'Multi-feature'
        
        # Calculate feature contributions
        for feature in columns:
            feature_zscore = abs((outliers[feature] - self.df[feature].mean()) / self.df[feature].std())
            outliers[f'{feature}_contribution'] = feature_zscore
            
        contribution_cols = [f'{f}_contribution' for f in columns]
        total_contributions = outliers[contribution_cols].sum(axis=1)
        for feature in columns:
            outliers[f'{feature}_percent'] = (outliers[f'{feature}_contribution'] / total_contributions * 100).round(2)
        
        return outliers