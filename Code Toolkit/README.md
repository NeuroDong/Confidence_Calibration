# About
A comprehensive code toolkit for confidence calibration. It provides codes for:
 - Data acquisition, see [CaliData](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData).
 - Calibration methods, see [CaliMethod](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod).
 - Calibration metrics, see [CaliMetric](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric).
 - Calibration visualization, see [CaliShow](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliShow).

# Included Code

 - [CaliData](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData)
   - [Real_Data](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData/Real_Data)
     - [Logit_Datasets_for_Neural_Networks.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData/Real_Data/Logit_Datasets_for_Neural_Networks.py)
     - [CREDIT.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData/Real_Data/CREDIT.py)
   - [Simulated_Data](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData/Simulated_Data)
     - [Binomial_Process_Sampling.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliData/Simulated_Data/Binomial_Process_Sampling.py)
 - [CaliMethod](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod)
    - [General_loss](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/General_loss)
      - [Soft_binned_ECE.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/General_loss/Soft_binned_ECE.py)
      - [Soft_AvUS.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/General_loss/Soft_AvUS.py)
      - [AvUS.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/General_loss/AvUS.py)
      - [DECE.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/General_loss/DECE.py)s
      - [dual_focal_loss.py](https://github.com/NeuroDong/Confidence_Calibration/blob/main/Code%20Toolkit/CaliMethod/General_loss/dual_focal_loss.py)
      - [LSECE.py](https://github.com/NeuroDong/Confidence_Calibration/blob/main/Code%20Toolkit/CaliMethod/General_loss/LSECE.py)
    - [In_training](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/In_training)
    - [Post_hoc](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/Post_hoc)
      - [scaling_binning.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/Post_hoc/scaling_binning.py)
      - [temperature_scaling.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMethod/temperature_scaling.py)
 - [CaliMetric](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric)
    - [Multi_Calibration](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric/Multi_Calibration)
      - [ECE_KDE.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric/Multi_Calibration/ECE_KDE.py)
    - [Top_Label_Calibration](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric/Top_Label_Calibration)
      - [Debiased_ECE.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric/Top_Label_Calibration/Debiased_ECE.py)
      - [ECE.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric/Top_Label_Calibration/ECE.py)
      - [KS_error.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliMetric/Top_Label_Calibration/KS_error.py)
 - [CaliShow](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliShow)
    - [validity_plot.py](https://github.com/NeuroDong/Confidence_Calibration/tree/main/Code%20Toolkit/CaliShow/validity_plot.py)