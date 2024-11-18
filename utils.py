# File: utils.py
import os  # Add this at the top
from datetime import datetime  # Add this too
import pandas as pd  # Add this for the save_to_excel function

def create_results_directory():
    """Create timestamped directory for results"""
    current_dir = os.getcwd()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(current_dir, f'analysis_results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_to_excel(results, results_dir):
    """Save all results to Excel file"""
    excel_path = os.path.join(results_dir, 'analysis_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Save correlation results
        if 'correlations' in results:
            results['correlations']['global_correlations'].to_excel(
                writer, sheet_name='Correlations')
            results['correlations']['trade_correlations'].to_excel(
                writer, sheet_name='Trade_Correlations')
        
        # Save outlier results
        if 'zscore_outliers' in results and not results['zscore_outliers'].empty:
            results['zscore_outliers'].to_excel(
                writer, sheet_name='ZScore_Outliers', index=False)
        
        if 'iqr_outliers' in results and not results['iqr_outliers'].empty:
            results['iqr_outliers'].to_excel(
                writer, sheet_name='IQR_Outliers', index=False)
        
        # Save isolation forest results
        if 'isolation_forest' in results:
            if not results['isolation_forest']['single_feature'].empty:
                results['isolation_forest']['single_feature'].to_excel(
                    writer, sheet_name='IF_Single_Feature', index=False)
            if not results['isolation_forest']['multi_feature'].empty:
                results['isolation_forest']['multi_feature'].to_excel(
                    writer, sheet_name='IF_Multi_Feature', index=False)

        # Save custom isolation forest results
        if 'custom_iforest' in results and not results['custom_iforest'].empty:
            results['custom_iforest'].to_excel(
                writer, sheet_name='Custom_IForest', index=False)

    
    print(f"\nResults saved to: {excel_path}")