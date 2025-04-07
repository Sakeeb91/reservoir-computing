import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set up the output directory
output_dir = Path("plots/ecg_anomalies")
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn')
sns.set_context("talk")

def create_ecg_plot(data, title, filename, annotations=None, ylim=(-1.5, 1.5)):
    """Create a standardized ECG plot with annotations."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data, linewidth=2)
    ax.set_title(title, pad=20)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    
    if annotations:
        for ann in annotations:
            ax.annotate(ann['text'], 
                       xy=ann['xy'], 
                       xytext=ann['xytext'],
                       arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def generate_mi_patterns():
    """Generate plots for myocardial infarction patterns."""
    # Acute MI pattern
    t = np.linspace(0, 1000, 1000)
    baseline = np.zeros_like(t)
    st_elevation = 0.3 * np.exp(-(t-500)**2/(2*50**2))
    mi_pattern = baseline + st_elevation
    
    annotations = [
        {'text': 'ST Elevation', 'xy': (500, 0.3), 'xytext': (400, 0.5)},
        {'text': 'Hyperacute T', 'xy': (600, 0.1), 'xytext': (650, 0.4)}
    ]
    
    create_ecg_plot(mi_pattern, "Acute Myocardial Infarction Pattern", 
                   "acute_mi_pattern.png", annotations)

def generate_arrhythmia_patterns():
    """Generate plots for arrhythmia patterns."""
    # Atrial Fibrillation pattern
    t = np.linspace(0, 2000, 2000)
    af_pattern = np.random.normal(0, 0.1, len(t))
    qrs_complexes = np.zeros_like(t)
    for i in range(0, len(t), 200):
        qrs_complexes[i:i+50] = 1.0 * np.exp(-(np.arange(50)-25)**2/(2*5**2))
    af_pattern += qrs_complexes
    
    annotations = [
        {'text': 'Irregular R-R', 'xy': (200, 1.0), 'xytext': (100, 1.2)},
        {'text': 'No P waves', 'xy': (500, 0.0), 'xytext': (400, 0.3)}
    ]
    
    create_ecg_plot(af_pattern, "Atrial Fibrillation Pattern", 
                   "af_pattern.png", annotations)

def generate_bundle_branch_patterns():
    """Generate plots for bundle branch block patterns."""
    # LBBB pattern
    t = np.linspace(0, 400, 400)
    lbbb_pattern = np.zeros_like(t)
    lbbb_pattern[100:150] = 1.0  # Wide QRS
    lbbb_pattern[150:200] = -0.5  # Deep S wave
    
    annotations = [
        {'text': 'Wide QRS', 'xy': (125, 1.0), 'xytext': (50, 1.2)},
        {'text': 'Deep S wave', 'xy': (175, -0.5), 'xytext': (200, -0.7)}
    ]
    
    create_ecg_plot(lbbb_pattern, "Left Bundle Branch Block Pattern", 
                   "lbbb_pattern.png", annotations)

def generate_premature_contraction_patterns():
    """Generate plots for premature contraction patterns."""
    # PVC pattern
    t = np.linspace(0, 800, 800)
    pvc_pattern = np.zeros_like(t)
    pvc_pattern[200:250] = 1.5  # Wide QRS complex
    pvc_pattern[250:300] = -0.8  # T wave
    
    annotations = [
        {'text': 'Wide QRS', 'xy': (225, 1.5), 'xytext': (150, 1.7)},
        {'text': 'No P wave', 'xy': (200, 0.0), 'xytext': (100, 0.3)}
    ]
    
    create_ecg_plot(pvc_pattern, "Premature Ventricular Contraction Pattern", 
                   "pvc_pattern.png", annotations)

def generate_st_t_wave_patterns():
    """Generate plots for ST-T wave abnormalities."""
    # Ischemic ST-T changes
    t = np.linspace(0, 600, 600)
    st_depression = np.zeros_like(t)
    st_depression[200:300] = -0.2  # ST depression
    st_depression[300:400] = 0.3  # T wave inversion
    
    annotations = [
        {'text': 'ST Depression', 'xy': (250, -0.2), 'xytext': (200, -0.4)},
        {'text': 'T wave inversion', 'xy': (350, 0.3), 'xytext': (400, 0.5)}
    ]
    
    create_ecg_plot(st_depression, "Ischemic ST-T Changes", 
                   "ischemic_st_t.png", annotations)

def generate_conduction_abnormality_patterns():
    """Generate plots for conduction abnormalities."""
    # AV Block pattern
    t = np.linspace(0, 1000, 1000)
    av_block = np.zeros_like(t)
    for i in range(0, len(t), 200):
        av_block[i:i+20] = 0.2  # P wave
        av_block[i+100:i+120] = 1.0  # QRS complex
    
    annotations = [
        {'text': 'P wave', 'xy': (100, 0.2), 'xytext': (50, 0.4)},
        {'text': 'QRS complex', 'xy': (200, 1.0), 'xytext': (150, 1.2)}
    ]
    
    create_ecg_plot(av_block, "AV Block Pattern", 
                   "av_block_progression.png", annotations)

def generate_model_performance_plots():
    """Generate plots for model performance."""
    # ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Example ROC curve
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve for Anomaly Detection")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance_roc.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all ECG anomaly plots."""
    print("Generating ECG anomaly plots...")
    
    generate_mi_patterns()
    generate_arrhythmia_patterns()
    generate_bundle_branch_patterns()
    generate_premature_contraction_patterns()
    generate_st_t_wave_patterns()
    generate_conduction_abnormality_patterns()
    generate_model_performance_plots()
    
    print(f"Plots generated in {output_dir}")

if __name__ == "__main__":
    main() 