# Hospital Patient Readmission Prediction

**Course:** AI for Software Engineering  
**Assignment:** Understanding the AI Development Workflow  

## Overview

This project implements an AI system to predict hospital patient readmission risk within 30 days of discharge. The implementation demonstrates the complete AI development workflow from problem definition to deployment.

## Files

- `preprocessing_pipeline.py` - Data preprocessing and feature engineering
- `confusion_matrix_analysis.py` - Model evaluation and performance metrics
- `workflow_diagrams.py` - AI development workflow visualizations
- `AI_Workflow_Research_Paper.pdf` - Complete research paper (8 pages)

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Usage

### 1. Run Data Preprocessing
Creates synthetic hospital data and performs feature engineering.

### 2. Run Model Evaluation
Generates confusion matrix and calculates performance metrics:
- Precision: 58.3%
- Recall: 70.0% 
- F1-Score: 63.6%

### 3. Generate Workflow Diagrams
Creates three workflow visualizations:
- Main AI Development Workflow (18 stages)
- Healthcare-Specific Implementation
- CRISP-DM Methodology

## Key Results

| Metric | Value | Meaning |
|--------|-------|---------|
| Accuracy | 92.0% | Overall correctness |
| Precision | 58.3% | Predicted readmissions that were correct |
| Recall | 70.0% | Actual readmissions correctly identified |
| Specificity | 94.4% | Non-readmissions correctly identified |

## Features

**Data Processing:**
- Synthetic hospital data (1000 patients)
- Missing value imputation
- Feature engineering (comorbidity indices, risk factors)
- Data quality assessment

**Workflow Documentation:**
- Complete 18-stage AI development process
- Healthcare-specific considerations
- HIPAA compliance and bias mitigation
- Regulatory requirements

## Assignment Components

✅ **Part 1: Short Answers** - Technical implementations  
✅ **Part 2: Case Study** - Hospital readmission prediction  
✅ **Part 3: Critical Thinking** - Ethics and bias analysis  
✅ **Part 4: Reflection** - Workflow diagrams and insights  

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost

## Output

The code generates:
- Processed dataset ready for modeling
- Performance metrics and visualizations
- Workflow diagrams (PNG format)
- Detailed analysis reports

## Healthcare Context

This implementation addresses real healthcare challenges:
- Patient safety through early readmission identification
- Resource optimization for hospital operations
- Regulatory compliance (HIPAA, FDA guidelines)
- Algorithmic fairness across patient demographics
