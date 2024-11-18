# File: correlation_analyzer.py
import os  # Add this at the top
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base_analyzer import BaseAnalyzer

class CorrelationAnalyzer(BaseAnalyzer):
    def analyze(self):
        print("\nAnalyzing correlations...")
        correlations = self.df[self.numerical_columns].corr()
        
        trade_correlations = {}
        for trade_id, group in self.df.groupby(self.id_column):
            trade_correlations[trade_id] = group[self.numerical_columns].corr()[self.target_column]
        
        self._create_heatmap(correlations)
        
        return {
            'global_correlations': correlations,
            'trade_correlations': pd.DataFrame(trade_correlations).T
        }
    
    def _create_heatmap(self, correlations):
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Global Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'global_correlations.png'))
        plt.close()