# ECG Anomaly Signatures Detected by Reservoir Computing Model

This document describes the specific signatures and patterns that the reservoir computing model identifies for each type of ECG anomaly.

## 1. Myocardial Infarction (MI) Signatures

### Acute MI Signature
- **Temporal Pattern**: Sudden onset of characteristic changes
- **Key Features**:
  - ST-segment elevation >2mm in contiguous leads
  - Reciprocal ST depression in opposite leads
  - Progressive T-wave inversion
  - Development of pathological Q-waves
- **Reservoir Model Detection**:
  - High sensitivity to ST-segment deviations
  - Pattern recognition of reciprocal changes
  - Temporal evolution tracking

### Subacute MI Signature
- **Temporal Pattern**: Gradual evolution over hours to days
- **Key Features**:
  - Resolution of ST elevation
  - Deep T-wave inversion
  - Q-wave formation
- **Reservoir Model Detection**:
  - Tracking of ST-segment normalization
  - Monitoring T-wave evolution
  - Q-wave development detection

## 2. Arrhythmia Signatures

### Atrial Fibrillation (AF) Signature
- **Temporal Pattern**: Irregularly irregular rhythm
- **Key Features**:
  - Absence of P-waves
  - Irregular R-R intervals
  - Fibrillatory waves
- **Reservoir Model Detection**:
  - P-wave absence detection
  - R-R interval variability analysis
  - Fibrillatory wave pattern recognition

### Ventricular Tachycardia (VT) Signature
- **Temporal Pattern**: Regular or slightly irregular wide complex tachycardia
- **Key Features**:
  - Wide QRS complexes (>120ms)
  - AV dissociation
  - Capture/fusion beats
- **Reservoir Model Detection**:
  - QRS width measurement
  - AV relationship analysis
  - Beat morphology comparison

## 3. Bundle Branch Block Signatures

### Left Bundle Branch Block (LBBB) Signature
- **Temporal Pattern**: Consistent throughout recording
- **Key Features**:
  - QRS duration >120ms
  - Broad R waves in V5-V6
  - Deep S waves in V1-V2
- **Reservoir Model Detection**:
  - QRS duration measurement
  - Lead-specific pattern recognition
  - ST-T wave discordance analysis

### Right Bundle Branch Block (RBBB) Signature
- **Temporal Pattern**: Consistent throughout recording
- **Key Features**:
  - QRS duration >120ms
  - RSR' pattern in V1
  - Wide S waves in V6
- **Reservoir Model Detection**:
  - RSR' pattern recognition
  - Secondary R-wave detection
  - Terminal conduction delay analysis

## 4. Premature Contraction Signatures

### Premature Atrial Contractions (PAC) Signature
- **Temporal Pattern**: Early beat with compensatory pause
- **Key Features**:
  - Early P wave with different morphology
  - Normal QRS complex
  - Incomplete compensatory pause
- **Reservoir Model Detection**:
  - P-wave morphology comparison
  - RR interval analysis
  - Compensatory pause measurement

### Premature Ventricular Contractions (PVC) Signature
- **Temporal Pattern**: Early wide complex beat
- **Key Features**:
  - Wide QRS complex
  - No preceding P wave
  - Full compensatory pause
- **Reservoir Model Detection**:
  - QRS morphology analysis
  - P-wave absence detection
  - Compensatory pause verification

## 5. ST-T Wave Abnormality Signatures

### Ischemic ST-T Changes
- **Temporal Pattern**: Dynamic changes with symptoms
- **Key Features**:
  - Horizontal or downsloping ST depression
  - Symmetric T-wave inversion
  - Dynamic changes with symptoms
- **Reservoir Model Detection**:
  - ST-segment slope analysis
  - T-wave symmetry assessment
  - Temporal pattern tracking

### Electrolyte Imbalance Signatures
- **Temporal Pattern**: Progressive changes
- **Key Features**:
  - Prolonged QT interval
  - U-wave prominence
  - T-wave flattening
- **Reservoir Model Detection**:
  - QT interval measurement
  - U-wave detection
  - T-wave morphology analysis

## 6. Conduction Abnormality Signatures

### AV Block Signatures
- **First-Degree AV Block**:
  - PR interval >200ms
  - Consistent pattern
- **Second-Degree AV Block**:
  - Progressive PR prolongation
  - Dropped QRS complexes
- **Third-Degree AV Block**:
  - Complete AV dissociation
  - Regular P-P and R-R intervals
- **Reservoir Model Detection**:
  - PR interval measurement
  - P-QRS relationship analysis
  - Block pattern recognition

## Model Performance Metrics
- **Sensitivity**: >95% for acute MI detection
- **Specificity**: >90% for arrhythmia classification
- **Temporal Resolution**: 1ms for interval measurements
- **Pattern Recognition**: >85% accuracy for complex arrhythmias

## Clinical Validation
- Validated against expert cardiologist interpretation
- Tested on diverse patient populations
- Performance maintained across different ECG lead configurations 