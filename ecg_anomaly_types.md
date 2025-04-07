# ECG Anomaly Types

This document describes the different types of anomalies found in the ECG Heartbeat Categorization Dataset.

## Dataset Overview
The dataset contains ECG signals from two main sources:
1. PTB Diagnostic ECG Database (ptbdb_normal.csv and ptbdb_abnormal.csv)
2. MIT-BIH Arrhythmia Database (mitbih_train.csv and mitbih_test.csv)

## Types of Anomalies

### 1. Myocardial Infarction (MI)
- **Description**: Heart attack caused by blockage of blood flow to the heart muscle
- **ECG Characteristics**:
  - ST-segment elevation or depression
  - T-wave inversion
  - Pathological Q-waves
  - Changes in R-wave progression

### 2. Arrhythmias
- **Description**: Irregular heart rhythms
- **Types**:
  - **Atrial Fibrillation (AF)**: Irregular and often rapid heart rate
  - **Ventricular Tachycardia (VT)**: Fast heart rate originating in the ventricles
  - **Bradycardia**: Abnormally slow heart rate
  - **Tachycardia**: Abnormally fast heart rate

### 3. Bundle Branch Blocks
- **Description**: Delay or blockage in the electrical conduction system of the heart
- **Types**:
  - **Left Bundle Branch Block (LBBB)**
  - **Right Bundle Branch Block (RBBB)**
  - **ECG Characteristics**: Widened QRS complex (>120ms)

### 4. Premature Contractions
- **Description**: Early heartbeats that originate in the atria or ventricles
- **Types**:
  - **Premature Atrial Contractions (PAC)**
  - **Premature Ventricular Contractions (PVC)**
  - **ECG Characteristics**: Early beats with abnormal morphology

### 5. ST-T Wave Abnormalities
- **Description**: Changes in the ST segment and T wave
- **Causes**:
  - Ischemia
  - Electrolyte imbalances
  - Drug effects
  - Myocardial disease

### 6. Conduction Abnormalities
- **Description**: Problems with the heart's electrical conduction system
- **Types**:
  - **First-Degree AV Block**
  - **Second-Degree AV Block**
  - **Third-Degree AV Block (Complete Heart Block)**

## Detection Methods
The reservoir computing model can detect these anomalies by:
1. Learning the normal ECG patterns
2. Identifying deviations from normal patterns
3. Classifying the type of anomaly based on characteristic features
4. Providing confidence scores for each detected anomaly

## Clinical Significance
Early detection of these anomalies is crucial for:
- Preventing cardiac events
- Guiding treatment decisions
- Monitoring disease progression
- Improving patient outcomes

## References
1. PTB Diagnostic ECG Database
2. MIT-BIH Arrhythmia Database
3. American Heart Association ECG Guidelines 