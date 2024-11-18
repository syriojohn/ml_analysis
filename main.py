# File: main.py
import pandas as pd
import os
import time
from correlation_analyzer import CorrelationAnalyzer
from outlier_analyzer import OutlierAnalyzer
from isolation_forest_analyzer import IsolationForestAnalyzer
from utils import create_results_directory, save_to_excel
from custom_iforest_analyzer import CustomIsolationForestAnalyzer

class DataAnalyzer:
    def __init__(self, file_path, date_column='ValuationDate', id_column='TradeId', 
                 numerical_columns=None, target_column=None):
        print(f"Loading data from: {file_path}")
        self.df = pd.read_csv(file_path)
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        self.date_column = date_column
        self.id_column = id_column
        
        # Auto-detect numerical columns if none specified
        if numerical_columns is None:
            self.numerical_columns = self.df.select_dtypes(
                include=['int64', 'float64']).columns.tolist()
        else:
            self.numerical_columns = list(numerical_columns)  # Convert to list if it's not already
            
        # Set target column
        self.target_column = target_column or self.numerical_columns[0]
        
        # Ensure target column is in numerical columns
        if self.target_column not in self.numerical_columns:
            self.numerical_columns.append(self.target_column)
        
        # Create results directory
        self.results_dir = create_results_directory()
        
        # Initialize analyzers
        self.correlation_analyzer = CorrelationAnalyzer(
            self.df, date_column, id_column, self.numerical_columns, 
            self.target_column, self.results_dir)
        self.outlier_analyzer = OutlierAnalyzer(
            self.df, date_column, id_column, self.numerical_columns, 
            self.target_column, self.results_dir)
        self.isolation_forest_analyzer = IsolationForestAnalyzer(
            self.df, date_column, id_column, self.numerical_columns, 
            self.target_column, self.results_dir)

        self.custom_iforest_analyzer = CustomIsolationForestAnalyzer(
            self.df, date_column, id_column, self.numerical_columns, 
            self.target_column, self.results_dir)

        print(f"\nAnalysis configuration:")
        print(f"Date column: {date_column}")
        print(f"ID column: {id_column}")
        print(f"Target column: {self.target_column}")
        print(f"Numerical columns: {', '.join(self.numerical_columns)}")

    def run_analysis(self):
        """Run all analyses and save results"""
        import time
        start_time = time.time()
    
        print("\nStarting comprehensive analysis...")
        print(f"Dataset size: {len(self.df):,} points")
    
        try:
            results = {
                'correlations': self.correlation_analyzer.analyze(),
                'zscore_outliers': self.outlier_analyzer.analyze_zscore(),
                'iqr_outliers': self.outlier_analyzer.analyze_iqr(),
                'isolation_forest': self.isolation_forest_analyzer.analyze(),
                'custom_iforest': self.custom_iforest_analyzer.analyze()
            }
        
            save_to_excel(results, self.results_dir)
        
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\nTotal analysis completed in {execution_time:.2f} seconds")
        
            return results
        
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Analysis failed after {execution_time:.2f} seconds")
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dynamic Data Analysis')
    parser.add_argument('--file', '-f', required=True, 
                       help='Path to the CSV file containing data')
    parser.add_argument('--date-column', default='ValuationDate',
                       help='Name of the date column')
    parser.add_argument('--id-column', default='TradeId',
                       help='Name of the ID column')
    parser.add_argument('--target-column', 
                       help='Primary column for single-feature analysis')
    parser.add_argument('--numerical-columns', nargs='+',
                       help='List of numerical columns to analyze')
    
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(
        args.file,
        date_column=args.date_column,
        id_column=args.id_column,
        numerical_columns=args.numerical_columns,
        target_column=args.target_column
    )
    analyzer.run_analysis()