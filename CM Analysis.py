"""
Confusion Matrix and Performance Metrics Calculator

This creates and analyzes confusion matrices for the hospital readmission prediction model,
calculating various performance metrics including precision, recall, F1-score, and ROC-AUC.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ConfusionMatrixAnalyzer:
    """
    Comprehensive confusion matrix analysis and visualization
    """
    
    def __init__(self):
        self.model = None
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def generate_hypothetical_predictions(self):
        """
        Generate hypothetical predictions for demonstration
        Based on the confusion matrix provided in the assignment
        """
        print("=== GENERATING HYPOTHETICAL PREDICTIONS ===")
        
        # Based on assignment's hypothetical confusion matrix:
        # Actual          No Readmission  Readmission
        # No Readmission      850           50
        # Readmission          30           70
        
        # True Negatives: 850, False Positives: 50
        # False Negatives: 30, True Positives: 70
        
        # Generate corresponding arrays
        tn, fp, fn, tp = 850, 50, 30, 70
        total = tn + fp + fn + tp
        
        # Create actual labels
        y_true = np.concatenate([
            np.zeros(tn + fp),     # No readmission cases
            np.ones(fn + tp)       # Readmission cases
        ])
        
        # Create predictions
        y_pred = np.concatenate([
            np.zeros(tn),          # Correctly predicted no readmission
            np.ones(fp),           # Incorrectly predicted readmission
            np.zeros(fn),          # Incorrectly predicted no readmission
            np.ones(tp)            # Correctly predicted readmission
        ])
        
        # Generate probability scores (for ROC analysis)
        np.random.seed(42)
        y_pred_proba = np.concatenate([
            np.random.beta(2, 5, tn),      # Low probabilities for TN
            np.random.beta(4, 3, fp),      # Higher probabilities for FP
            np.random.beta(3, 4, fn),      # Lower probabilities for FN
            np.random.beta(5, 2, tp)       # High probabilities for TP
        ])
        
        self.y_true = y_true.astype(int)
        self.y_pred = y_pred.astype(int)
        self.y_pred_proba = y_pred_proba
        
        print(f"Generated {total} predictions")
        print(f"Actual distribution: No readmission={np.sum(y_true==0)}, Readmission={np.sum(y_true==1)}")
        print(f"Predicted distribution: No readmission={np.sum(y_pred==0)}, Readmission={np.sum(y_pred==1)}")
        
        return self.y_true, self.y_pred, self.y_pred_proba
    
    def calculate_confusion_matrix(self, y_true=None, y_pred=None):
        """
        Calculate and display confusion matrix
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
            
        print("\n=== CONFUSION MATRIX ANALYSIS ===")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        print("Confusion Matrix:")
        print(f"                    Predicted")
        print(f"Actual          No Readmission  Readmission")
        print(f"No Readmission      {tn:4d}         {fp:4d}")
        print(f"Readmission         {fn:4d}          {tp:4d}")
        
        return cm, tn, fp, fn, tp
    
    def calculate_performance_metrics(self, tn, fp, fn, tp):
        """
        Calculate comprehensive performance metrics
        """
        print("\n=== PERFORMANCE METRICS ===")
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Calculate ROC-AUC if probability scores available
        roc_auc = None
        if self.y_pred_proba is not None:
            roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'roc_auc': roc_auc
        }
        
        # Display metrics
        print(f"Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Precision:   {precision:.3f} ({precision*100:.1f}%)")
        print(f"Recall:      {recall:.3f} ({recall*100:.1f}%)")
        print(f"Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"F1-Score:    {f1_score:.3f}")
        print(f"NPV:         {npv:.3f} ({npv*100:.1f}%)")
        print(f"FPR:         {fpr:.3f} ({fpr*100:.1f}%)")
        print(f"FNR:         {fnr:.3f} ({fnr*100:.1f}%)")
        if roc_auc:
            print(f"ROC-AUC:     {roc_auc:.3f}")
        
        return metrics
    
    def interpret_metrics(self, metrics):
        """
        Provide interpretation of the metrics in healthcare context
        """
        print("\n=== METRICS INTERPRETATION FOR HEALTHCARE ===")
        
        precision = metrics['precision']
        recall = metrics['recall']
        specificity = metrics['specificity']
        f1_score = metrics['f1_score']
        
        print("Healthcare Context Interpretation:")
        print(f"• Precision ({precision:.3f}): Of all patients predicted to be readmitted,")
        print(f"  {precision*100:.1f}% actually were readmitted. This affects resource allocation efficiency.")
        
        print(f"• Recall/Sensitivity ({recall:.3f}): Of all patients who were actually readmitted,")
        print(f"  {recall*100:.1f}% were correctly identified. This is critical for patient safety.")
        
        print(f"• Specificity ({specificity:.3f}): Of all patients who were not readmitted,")
        print(f"  {specificity*100:.1f}% were correctly identified as low-risk.")
        
        print(f"• F1-Score ({f1_score:.3f}): Balanced measure of precision and recall.")
        
        # Clinical recommendations
        print("\nClinical Recommendations:")
        if recall < 0.8:
            print("⚠️  Low recall: Many high-risk patients may be missed. Consider:")
            print("   - Lowering prediction threshold")
            print("   - Adding more sensitive features")
            print("   - Implementing additional screening protocols")
        
        if precision < 0.6:
            print("⚠️  Low precision: Many false alarms may waste resources. Consider:")
            print("   - Raising prediction threshold")
            print("   - Improving feature quality")
            print("   - Implementing tiered intervention strategies")
        
        if specificity > 0.9 and recall > 0.7:
            print("✅ Good balance: Model shows strong discriminative ability")
    
    def visualize_confusion_matrix(self, cm):
        """
        Create visualization of confusion matrix
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Readmission', 'Readmission'],
                   yticklabels=['No Readmission', 'Readmission'])
        
        plt.title('Confusion Matrix - Hospital Readmission Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        
        # Add percentage annotations
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=12, color='red')
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def plot_roc_curve(self):
        """
        Plot ROC curve if probability scores are available
        """
        if self.y_pred_proba is None:
            print("No probability scores available for ROC curve")
            return
            
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Hospital Readmission Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def generate_classification_report(self):
        """
        Generate detailed classification report
        """
        print("\n=== DETAILED CLASSIFICATION REPORT ===")
        
        target_names = ['No Readmission', 'Readmission']
        report = classification_report(self.y_true, self.y_pred, 
                                     target_names=target_names, 
                                     digits=3)
        print(report)
        
        # Convert to DataFrame for better presentation
        report_dict = classification_report(self.y_true, self.y_pred, 
                                          target_names=target_names, 
                                          output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        return report_df
    
    def run_complete_analysis(self):
        """
        Run complete confusion matrix analysis
        """
        print("CONFUSION MATRIX ANALYSIS - HOSPITAL READMISSION PREDICTION")
        print("=" * 70)
        
        # Step 1: Generate hypothetical predictions
        self.generate_hypothetical_predictions()
        
        # Step 2: Calculate confusion matrix
        cm, tn, fp, fn, tp = self.calculate_confusion_matrix()
        
        # Step 3: Calculate performance metrics
        metrics = self.calculate_performance_metrics(tn, fp, fn, tp)
        
        # Step 4: Interpret metrics
        self.interpret_metrics(metrics)
        
        # Step 5: Generate classification report
        report_df = self.generate_classification_report()
        
        # Step 6: Create visualizations
        print("\n=== CREATING VISUALIZATIONS ===")
        cm_fig = self.visualize_confusion_matrix(cm)
        roc_fig = self.plot_roc_curve()
        
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX ANALYSIS COMPLETED!")
        print("=" * 70)
        
        return metrics, report_df, cm

# Example usage and advanced analysis
def compare_models_example():
    """
    Example of comparing different model thresholds
    """
    print("\n=== MODEL THRESHOLD COMPARISON ===")
    
    analyzer = ConfusionMatrixAnalyzer()
    analyzer.generate_hypothetical_predictions()
    
    # Compare different thresholds
    thresholds = [0.3, 0.5, 0.7]
    results = []
    
    for threshold in thresholds:
        # Adjust predictions based on threshold
        y_pred_thresh = (analyzer.y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(analyzer.y_true, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
        
        print(f"Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConfusionMatrixAnalyzer()
    
    # Run complete analysis
    metrics, report_df, cm = analyzer.run_complete_analysis()
    
    # Run threshold comparison
    threshold_results = compare_models_example()
    
    print("\nAnalysis complete! Key insights:")
    print(f"• Model achieves {metrics['accuracy']*100:.1f}% overall accuracy")
    print(f"• Precision: {metrics['precision']*100:.1f}% - resource allocation efficiency")
    print(f"• Recall: {metrics['recall']*100:.1f}% - patient safety coverage")
    print(f"• F1-Score: {metrics['f1_score']:.3f} - balanced performance measure")