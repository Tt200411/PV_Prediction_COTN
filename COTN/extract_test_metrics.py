#!/usr/bin/env python3
"""
Extract and compile test metrics from test_results folder
Generates a comprehensive metrics summary in txt format
"""

import os
import json
import numpy as np
from datetime import datetime

def load_metrics_from_npy(results_dir):
    """Load basic metrics from numpy files"""
    try:
        metrics_path = os.path.join(results_dir, 'metrics.npy')
        if os.path.exists(metrics_path):
            metrics_array = np.load(metrics_path)
            return {
                'MAE': float(metrics_array[0]),
                'MSE': float(metrics_array[1]), 
                'RMSE': float(metrics_array[2]),
                'MAPE': float(metrics_array[3]),
                'MSPE': float(metrics_array[4])
            }
    except Exception as e:
        print(f"Warning: Could not load metrics.npy from {results_dir}: {e}")
    return None

def load_comprehensive_metrics_from_json(results_dir):
    """Load comprehensive metrics from JSON export files"""
    try:
        # Look for JSON files in test_exports directory
        json_files = []
        export_dir = './test_exports'
        if os.path.exists(export_dir):
            for file in os.listdir(export_dir):
                if file.endswith('_test_results.json'):
                    json_files.append(os.path.join(export_dir, file))
        
        # Also check in the results directory itself
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                json_files.append(os.path.join(results_dir, file))
        
        if json_files:
            # Use the first JSON file found
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('metrics', {})
    except Exception as e:
        print(f"Warning: Could not load JSON metrics from {results_dir}: {e}")
    return None

def generate_metrics_summary():
    """Generate comprehensive metrics summary from test_results folder"""
    
    test_results_dir = './test_results'
    if not os.path.exists(test_results_dir):
        print(f"Error: {test_results_dir} directory not found!")
        return
    
    # Find all result folders
    result_folders = []
    for item in os.listdir(test_results_dir):
        item_path = os.path.join(test_results_dir, item)
        if os.path.isdir(item_path):
            result_folders.append(item_path)
    
    if not result_folders:
        print(f"No result folders found in {test_results_dir}")
        return
    
    print(f"Found {len(result_folders)} result folders:")
    for folder in result_folders:
        print(f"  - {os.path.basename(folder)}")
    
    # Generate summary text
    summary_content = []
    summary_content.append("=" * 80)
    summary_content.append("COTN Solar Power Prediction - Test Results Summary")
    summary_content.append("=" * 80)
    summary_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_content.append(f"Total Datasets Tested: {len(result_folders)}")
    summary_content.append("=" * 80)
    summary_content.append("")
    
    all_metrics = []
    
    for folder_path in sorted(result_folders):
        folder_name = os.path.basename(folder_path)
        summary_content.append(f"Dataset: {folder_name}")
        summary_content.append("-" * 60)
        
        # Try to load comprehensive metrics first (from JSON)
        comprehensive_metrics = load_comprehensive_metrics_from_json(folder_path)
        
        # Fallback to basic metrics (from numpy)
        basic_metrics = load_metrics_from_npy(folder_path)
        
        if comprehensive_metrics:
            # Use comprehensive metrics
            metrics = comprehensive_metrics
            summary_content.append("📊 Comprehensive Metrics:")
            summary_content.append(f"   MAE (Mean Absolute Error):           {metrics.get('MAE', 'N/A'):.6f}")
            summary_content.append(f"   MSE (Mean Squared Error):            {metrics.get('MSE', 'N/A'):.6f}")
            summary_content.append(f"   RMSE (Root Mean Squared Error):      {metrics.get('RMSE', 'N/A'):.6f}")
            summary_content.append(f"   MAPE (Mean Absolute Percentage Error): {metrics.get('MAPE', 'N/A'):.4f}%")
            summary_content.append(f"   MSPE (Mean Squared Percentage Error):  {metrics.get('MSPE', 'N/A'):.6f}")
            summary_content.append(f"   R² (Coefficient of Determination):   {metrics.get('R2', 'N/A'):.6f}")
            summary_content.append(f"   SMAPE (Symmetric MAPE):              {metrics.get('SMAPE', 'N/A'):.4f}%")
            summary_content.append(f"   WAPE (Weighted Absolute Percentage Error): {metrics.get('WAPE', 'N/A'):.4f}%")
            summary_content.append(f"   NRMSE (Normalized RMSE):             {metrics.get('NRMSE', 'N/A'):.6f}")
            summary_content.append(f"   MBE (Mean Bias Error):               {metrics.get('MBE', 'N/A'):.6f}")
            summary_content.append(f"   RSE (Relative Standard Error):       {metrics.get('RSE', 'N/A'):.6f}")
            summary_content.append(f"   CORR (Correlation Coefficient):      {metrics.get('CORR', 'N/A'):.6f}")
            
        elif basic_metrics:
            # Use basic metrics only
            metrics = basic_metrics
            summary_content.append("📊 Basic Metrics:")
            summary_content.append(f"   MAE (Mean Absolute Error):           {metrics.get('MAE', 'N/A'):.6f}")
            summary_content.append(f"   MSE (Mean Squared Error):            {metrics.get('MSE', 'N/A'):.6f}")
            summary_content.append(f"   RMSE (Root Mean Squared Error):      {metrics.get('RMSE', 'N/A'):.6f}")
            summary_content.append(f"   MAPE (Mean Absolute Percentage Error): {metrics.get('MAPE', 'N/A'):.6f}")
            summary_content.append(f"   MSPE (Mean Squared Percentage Error):  {metrics.get('MSPE', 'N/A'):.6f}")
            
        else:
            summary_content.append("⚠️  No metrics found for this dataset")
            metrics = {}
        
        # Add to overall collection for summary statistics
        if metrics:
            metrics['dataset'] = folder_name
            all_metrics.append(metrics)
        
        summary_content.append("")
    
    # Add overall summary statistics
    if all_metrics:
        summary_content.append("=" * 80)
        summary_content.append("OVERALL SUMMARY STATISTICS")
        summary_content.append("=" * 80)
        
        # Calculate averages for common metrics
        common_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
        if all(metric in all_metrics[0] for metric in ['R2', 'CORR']):
            common_metrics.extend(['R2', 'SMAPE', 'WAPE', 'NRMSE', 'MBE', 'RSE', 'CORR'])
        
        summary_content.append(f"Average Performance Across {len(all_metrics)} Datasets:")
        summary_content.append("-" * 50)
        
        for metric in common_metrics:
            values = [m[metric] for m in all_metrics if metric in m and m[metric] != 'N/A']
            if values:
                avg_val = np.mean(values)
                std_val = np.std(values)
                if metric in ['MAPE', 'SMAPE', 'WAPE']:
                    summary_content.append(f"   {metric:40} : {avg_val:8.4f}% ± {std_val:.4f}%")
                else:
                    summary_content.append(f"   {metric:40} : {avg_val:8.6f} ± {std_val:.6f}")
        
        # Best and worst performing datasets
        if 'MAE' in all_metrics[0]:
            best_mae = min(all_metrics, key=lambda x: x.get('MAE', float('inf')))
            worst_mae = max(all_metrics, key=lambda x: x.get('MAE', 0))
            
            summary_content.append("")
            summary_content.append("Best/Worst Performance (by MAE):")
            summary_content.append("-" * 35)
            summary_content.append(f"   Best:  {best_mae['dataset']} (MAE: {best_mae['MAE']:.6f})")
            summary_content.append(f"   Worst: {worst_mae['dataset']} (MAE: {worst_mae['MAE']:.6f})")
    
    # Save to file
    output_file = f"test_metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_content))
    
    print(f"\n✅ Metrics summary saved to: {output_file}")
    print(f"📊 Processed {len(result_folders)} datasets")
    if all_metrics:
        avg_mae = np.mean([m.get('MAE', 0) for m in all_metrics if 'MAE' in m])
        print(f"📈 Average MAE across all datasets: {avg_mae:.6f}")
    
    return output_file

if __name__ == "__main__":
    print("COTN Test Results Metrics Extractor")
    print("=" * 50)
    generate_metrics_summary()