import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.metrics import comprehensive_metrics

# Set English font to avoid font issues
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_prediction_comparison(pred_data, true_data, dataset_name, setting, 
                              output_dir="./test_plots", show_samples=5, pred_len=100):
    """
    Plot prediction vs true values comparison for test results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate comprehensive metrics
    metrics = comprehensive_metrics(pred_data, true_data)
    
    # Select samples to display
    total_samples = pred_data.shape[0]
    if show_samples > total_samples:
        show_samples = total_samples
    
    sample_indices = np.linspace(0, total_samples-1, show_samples, dtype=int)
    
    # Create subplots
    fig, axes = plt.subplots(show_samples, 1, figsize=(15, 3*show_samples))
    if show_samples == 1:
        axes = [axes]
    
    fig.suptitle(f'{dataset_name} - Prediction Results\n'
                f'{setting}\n'
                f'MAE: {metrics["MAE"]:.4f}, RMSE: {metrics["RMSE"]:.4f}, '
                f'R²: {metrics["R2"]:.4f}', 
                fontsize=14, fontweight='bold')
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Get current sample predictions and true values
        pred_sample = pred_data[sample_idx, :, 0]  # Assume single feature
        true_sample = true_data[sample_idx, :, 0]
        
        # Create time axis
        x_axis = range(pred_len)
        
        # Plot predictions and true values
        ax.plot(x_axis, true_sample, label='True Values', color='blue', 
                linewidth=2, alpha=0.8)
        ax.plot(x_axis, pred_sample, label='Predictions', color='red', 
                linewidth=2, alpha=0.8, linestyle='--')
        
        # Calculate sample-specific metrics
        sample_mae = np.mean(np.abs(pred_sample - true_sample))
        sample_rmse = np.sqrt(np.mean((pred_sample - true_sample)**2))
        
        # Set plot properties
        ax.set_title(f'Sample {sample_idx+1} (MAE: {sample_mae:.4f}, RMSE: {sample_rmse:.4f})', 
                    fontsize=12)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Power Value (MW)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add error filling
        ax.fill_between(x_axis, true_sample, pred_sample, alpha=0.2, 
                       color='gray', label='Error Region')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{dataset_name}_{setting}_prediction_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prediction comparison plot saved: {plot_path}")
    return plot_path


def plot_comprehensive_metrics(pred_data, true_data, dataset_name, setting, 
                              output_dir="./test_plots"):
    """
    Plot comprehensive metrics analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate comprehensive metrics
    metrics = comprehensive_metrics(pred_data, true_data)
    
    # Create comprehensive metrics visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Error metrics bar chart
    ax1 = plt.subplot(2, 3, 1)
    error_metrics = ['MAE', 'MSE', 'RMSE', 'NRMSE']
    error_values = [metrics[m] for m in error_metrics]
    bars1 = ax1.bar(error_metrics, error_values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
    ax1.set_title('Error Metrics')
    ax1.set_ylabel('Value')
    for bar, val in zip(bars1, error_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom')
    
    # 2. Percentage error metrics
    ax2 = plt.subplot(2, 3, 2)
    perc_metrics = ['MAPE', 'SMAPE', 'WAPE']
    perc_values = [metrics[m] for m in perc_metrics]
    bars2 = ax2.bar(perc_metrics, perc_values, color=['gold', 'purple', 'brown'])
    ax2.set_title('Percentage Error Metrics')
    ax2.set_ylabel('Percentage (%)')
    for bar, val in zip(bars2, perc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perc_values)*0.01,
                f'{val:.2f}%', ha='center', va='bottom')
    
    # 3. Quality metrics
    ax3 = plt.subplot(2, 3, 3)
    quality_metrics = ['R²', 'CORR']
    quality_values = [metrics[m] for m in quality_metrics]
    bars3 = ax3.bar(quality_metrics, quality_values, color=['green', 'teal'])
    ax3.set_title('Quality Metrics')
    ax3.set_ylabel('Value')
    ax3.set_ylim(0, 1)
    for bar, val in zip(bars3, quality_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom')
    
    # 4. Error distribution histogram
    ax4 = plt.subplot(2, 3, 4)
    errors = (pred_data - true_data).flatten()
    ax4.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Prediction Error Distribution')
    ax4.set_xlabel('Error Value')
    ax4.set_ylabel('Frequency')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, 
               label=f'Mean Error: {np.mean(errors):.4f}')
    ax4.legend()
    
    # 5. Prediction vs True scatter plot
    ax5 = plt.subplot(2, 3, 5)
    pred_flat = pred_data.flatten()[:10000]  # Sample for visualization
    true_flat = true_data.flatten()[:10000]
    ax5.scatter(true_flat, pred_flat, alpha=0.3, s=1)
    min_val, max_val = min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax5.set_xlabel('True Values')
    ax5.set_ylabel('Predicted Values')
    ax5.set_title('Prediction vs True Values')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    metrics_table_data = []
    for metric_name, value in metrics.items():
        if metric_name in ['MAPE', 'SMAPE', 'WAPE']:
            metrics_table_data.append([metric_name, f'{value:.2f}%'])
        else:
            metrics_table_data.append([metric_name, f'{value:.4f}'])
    
    table = ax6.table(cellText=metrics_table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Comprehensive Metrics Summary', pad=20)
    
    plt.suptitle(f'{dataset_name} - {setting}\nComprehensive Evaluation Results', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    metrics_filename = f"{dataset_name}_{setting}_comprehensive_metrics.png"
    metrics_path = os.path.join(output_dir, metrics_filename)
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive metrics plot saved: {metrics_path}")
    return metrics_path


def save_test_results(pred_data, true_data, dataset_name, setting, 
                     output_dir="./test_results"):
    """
    Save test results in various formats with comprehensive metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate comprehensive metrics
    metrics = comprehensive_metrics(pred_data, true_data)
    
    # Convert all metrics to float to avoid JSON serialization issues
    clean_metrics = {}
    for key, value in metrics.items():
        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    clean_metrics[key] = float(value.item())
                else:
                    clean_metrics[key] = float(np.mean(value))
            elif hasattr(value, 'item'):
                clean_metrics[key] = float(value.item())
            else:
                clean_metrics[key] = float(value)
        except (ValueError, TypeError):
            clean_metrics[key] = float(np.mean(np.array(value).flatten()))
    
    # Prepare results dictionary
    results = {
        'dataset_info': {
            'dataset_name': dataset_name,
            'setting': setting,
            'total_samples': int(pred_data.shape[0]),
            'prediction_length': int(pred_data.shape[1]),
            'features': int(pred_data.shape[2])
        },
        'metrics': clean_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save JSON summary with comprehensive metrics
    json_filename = f"{dataset_name}_{setting}_test_results.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Test results JSON saved: {json_path}")
    
    # Save detailed CSV with sample-level metrics
    csv_data = []
    n_samples = pred_data.shape[0]
    
    for sample_idx in range(n_samples):
        pred_sample = pred_data[sample_idx]
        true_sample = true_data[sample_idx]
        
        # Calculate sample-level metrics
        sample_mae = np.mean(np.abs(pred_sample - true_sample))
        sample_mse = np.mean((pred_sample - true_sample)**2)
        sample_rmse = np.sqrt(sample_mse)
        sample_r2 = 1 - (np.sum((true_sample - pred_sample)**2) / 
                        np.sum((true_sample - true_sample.mean())**2))
        
        csv_data.append({
            'dataset': dataset_name,
            'setting': setting,
            'sample_id': sample_idx,
            'sample_mae': sample_mae,
            'sample_mse': sample_mse, 
            'sample_rmse': sample_rmse,
            'sample_r2': sample_r2,
            'pred_mean': float(np.mean(pred_sample)),
            'pred_std': float(np.std(pred_sample)),
            'pred_min': float(np.min(pred_sample)),
            'pred_max': float(np.max(pred_sample)),
            'true_mean': float(np.mean(true_sample)),
            'true_std': float(np.std(true_sample)),
            'true_min': float(np.min(true_sample)),
            'true_max': float(np.max(true_sample))
        })
    
    df = pd.DataFrame(csv_data)
    csv_filename = f"{dataset_name}_{setting}_sample_analysis.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"✅ Sample analysis CSV saved: {csv_path}")
    
    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'results': results
    }


def generate_test_report(dataset_name, setting, results_dir, plots_dir, export_dir):
    """
    Generate comprehensive test report in English
    """
    report_content = f"""# {dataset_name} Test Results Report

## Experiment Configuration
- Dataset: {dataset_name}
- Setting: {setting}
- Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## File Structure
- Test Results: `{results_dir}/`
- Visualization Plots: `{plots_dir}/`
- Export Files: `{export_dir}/`

## Generated Files
1. **Prediction Comparison Plot**: `{dataset_name}_{setting}_prediction_comparison.png`
   - Shows predicted vs true values for selected samples
   - Displays sample-level MAE and RMSE
   - Includes error region visualization

2. **Comprehensive Metrics Analysis**: `{dataset_name}_{setting}_comprehensive_metrics.png`  
   - Error metrics (MAE, MSE, RMSE, NRMSE)
   - Percentage errors (MAPE, SMAPE, WAPE)
   - Quality metrics (R², Correlation)
   - Error distribution and scatter plots
   - Complete metrics summary table

3. **Test Results Summary**: `{dataset_name}_{setting}_test_results.json`
   - Complete metrics evaluation
   - Dataset metadata
   - Timestamp information

4. **Sample Analysis**: `{dataset_name}_{setting}_sample_analysis.csv`
   - Per-sample performance metrics
   - Statistical summaries of predictions and true values
   - Ready for further statistical analysis

## Metrics Overview
The comprehensive evaluation includes:
- **Error Metrics**: MAE, MSE, RMSE, NRMSE
- **Percentage Errors**: MAPE, SMAPE, WAPE  
- **Quality Metrics**: R², Correlation coefficient
- **Bias Analysis**: Mean Bias Error (MBE)
- **Relative Errors**: Relative Standard Error (RSE)

## Usage Instructions
1. Review prediction comparison plots for visual assessment
2. Check comprehensive metrics for quantitative evaluation
3. Use JSON files for automated reporting and analysis
4. Analyze CSV files for detailed sample-level insights

---
Generated by COTN Solar Power Prediction Framework
Test-only mode with checkpoint loading
"""
    
    report_filename = f"{dataset_name}_{setting}_test_report.md"
    report_path = os.path.join(export_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Test report saved: {report_path}")
    return report_path