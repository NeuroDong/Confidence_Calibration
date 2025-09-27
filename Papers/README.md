# 📕 Table of Contents
- [📕 Table of Contents](#-table-of-contents)
- [Calibration Metric](#calibration-metric)
- [Calibration Method](#calibration-method)
  - [Train-time Calibration](#train-time-calibration)
  - [Post-hoc Calibration](#post-hoc-calibration)
    - [Parametric Method](#parametric-method)
    - [Non-parametric Method](#non-parametric-method)
    - [Mixed Method](#mixed-method)
- [Calibration under Distribution Shift](#calibration-under-distribution-shift)
- [Calibration under Distribution Shift](#calibration-under-distribution-shift-1)
  - [Calibration under Label Shift](#calibration-under-label-shift)
  - [Calibration under Covariate Shift](#calibration-under-covariate-shift)
  - [Calibration under Joint Shift](#calibration-under-joint-shift)
- [How to Update Citations](#how-to-update-citations)

# Calibration Metric
| Paper | Source | Year| Code| Citations |
|-------|-------|-------|-------|:--:|
|[Combining Priors with Experience: Confidence Calibration Based on Binomial Process Modeling](https://arxiv.org/abs/2412.10658)|AAAI|2025|[![Star](https://img.shields.io/github/stars/NeuroDong/TCEbpm.svg?style=social&label=Star)](https://github.com/NeuroDong/TCEbpm)| [![Citations](badges/Combining_Priors_with_Experience__Confidence_Calibration_Based_on_Binomial_Process_Modeling.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Combining+Priors+with+Experience%3A+Confidence+Calibration+Based+on+Binomial+Process+Modeling&btnG=) |
|[Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing](https://arxiv.org/abs/2309.12236)|ICLR|2024|[![Star](https://img.shields.io/github/stars/apple/ml-calibration.svg?style=social&label=Star)](https://github.com/apple/ml-calibration)| [![Citations](badges/Smooth_ECE__Principled_Reliability_Diagrams_via_Kernel_Smoothing.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Smooth+ECE%3A+Principled+Reliability+Diagrams+via+Kernel+Smoothing&btnG=) |
|[Proximity-Informed Calibration for Deep Neural Networks](https://arxiv.org/abs/2306.04590)|NeurIPS|2023|[![Star](https://img.shields.io/github/stars/MiaoXiong2320/ProximityBias-Calibration.svg?style=social&label=Star)](https://github.com/MiaoXiong2320/ProximityBias-Calibration)| [![Citations](badges/Proximity-Informed_Calibration_for_Deep_Neural_Networks.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Proximity-Informed+Calibration+for+Deep+Neural+Networks&btnG=) |
|[A Unifying Theory of Distance from Calibration](https://arxiv.org/abs/2211.16886)|STOC|2023|None| [![Citations](badges/A_Unifying_Theory_of_Distance_from_Calibration.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=A+Unifying+Theory+of+Distance+from+Calibration&btnG=) |
|[Beyond calibration: estimating the grouping loss of modern neural networks](https://arxiv.org/abs/2210.16315)|ICLR|2023|[![Star](https://img.shields.io/github/stars/aperezlebel/beyond_calibration.svg?style=social&label=Star)](https://github.com/aperezlebel/beyond_calibration)| [![Citations](badges/Beyond_calibration__estimating_the_grouping_loss_of_modern_neural_networks.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Beyond+calibration%3A+estimating+the+grouping+loss+of+modern+neural+networks&btnG=) |
|[Consistent and Asymptotically Unbiased Estimation of Proper Calibration Errors](https://arxiv.org/abs/2312.08589)|AISTATS|2023|[![Star](https://img.shields.io/github/stars/tpopordanoska/proper-calibration-error.svg?style=social&label=Star)](https://github.com/tpopordanoska/proper-calibration-error)| [![Citations](badges/Consistent_and_Asymptotically_Unbiased_Estimation_of_Proper_Calibration_Errors.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Consistent+and+Asymptotically+Unbiased+Estimation+of+Proper+Calibration+Errors&btnG=) |
|[A Consistent and Differentiable Lp Canonical Calibration Error Estimator](https://arxiv.org/abs/2210.07810)|NeurIPS|2022|[![Star](https://img.shields.io/github/stars/tpopordanoska/ece-kde.svg?style=social&label=Star)](https://github.com/tpopordanoska/ece-kde)| [![Citations](badges/A_Consistent_and_Differentiable_Lp_Canonical_Calibration_Error_Estimator.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=A+Consistent+and+Differentiable+Lp+Canonical+Calibration+Error+Estimator&btnG=) |
|[T-Cal: An optimal test for the calibration of predictive models](https://arxiv.org/abs/2203.01850)|JMLR|2022|[![Star](https://img.shields.io/github/stars/dh7401/T-Cal.svg?style=social&label=Star)](https://github.com/dh7401/T-Cal)| [![Citations](badges/T-Cal__An_optimal_test_for_the_calibration_of_predictive_models.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=T-Cal%3A+An+optimal+test+for+the+calibration+of+predictive+models&btnG=) |
|[Metrics of calibration for probabilistic predictions](https://arxiv.org/abs/2205.09680)|JMLR|2022|[![Star](https://img.shields.io/github/stars/facebookresearch/ecevecce.svg?style=social&label=Star)](https://github.com/facebookresearch/ecevecce)| [![Citations](badges/Metrics_of_calibration_for_probabilistic_predictions.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Metrics+of+calibration+for+probabilistic+predictions&btnG=) |
|[Mitigating Bias in Calibration Error Estimation](https://proceedings.mlr.press/v151/roelofs22a.html)|AISTATS|2022|[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/google-research/google-research/tree/master/caltrain)| [![Citations](badges/Mitigating_Bias_in_Calibration_Error_Estimation.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Mitigating+Bias+in+Calibration+Error+Estimation&btnG=) |
|[Stable reliability diagrams for probabilistic classifiers](https://www.pnas.org/doi/abs/10.1073/pnas.2016191118)|PNAS|2021|[![Star](https://img.shields.io/github/stars/TimoDimi/replication_DGJ20.svg?style=social&label=Star)](https://github.com/TimoDimi/replication_DGJ20)|[![Citations](badges/Stable_reliability_diagrams_for_probabilistic_classifiers.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Stable+reliability+diagrams+for+probabilistic+classifiers&btnG=)|
|[Distribution-Free Calibration Guarantees for Histogram Binning without Sample Splitting](https://proceedings.mlr.press/v139/gupta21b.html)|ICML|2021|[![Star](https://img.shields.io/github/stars/aigen/df-posthoc-calibration.svg?style=social&label=Star)](https://github.com/aigen/df-posthoc-calibration)| [![Citations](badges/Distribution-Free_Calibration_Guarantees_for_Histogram_Binning_without_Sample_Splitting.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Distribution-Free+Calibration+Guarantees+for+Histogram+Binning+without+Sample+Splitting&btnG=) |
|[Calibration of Neural Networks using Splines](https://arxiv.org/abs/2006.12800)|ICLR|2021|[![Star](https://img.shields.io/github/stars/kartikgupta-at-anu/spline-calibration.svg?style=social&label=Star)](https://github.com/kartikgupta-at-anu/spline-calibration)| [![Citations](badges/Calibration_of_Neural_Networks_using_Splines.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Calibration+of+Neural+Networks+using+Splines&btnG=) | 
|[Evaluating model calibration in classification](https://proceedings.mlr.press/v89/vaicenavicius19a.html)|AISTATS|2019|[![Star](https://img.shields.io/github/stars/uu-sml/calibration.svg?style=social&label=Star)](https://github.com/uu-sml/calibration)| [![Citations](badges/Evaluating_model_calibration_in_classification.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Evaluating+model+calibration+in+classification+Vaicenavicius&btnG=) |
|[Verified Uncertainty Calibration](https://proceedings.neurips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)|NeurIPS|2019|[![Star](https://img.shields.io/github/stars/p-lambda/verified_calibration.svg?style=social&label=Star)](https://github.com/p-lambda/verified_calibration)| [![Citations](badges/Verified_Uncertainty_Calibration.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Verified+Uncertainty+Calibration&btnG=) | 


# Calibration Method
## Train-time Calibration
| Paper | Name| Source | Year| Code| Citations |
|-------|-------|-------|-------|-------|:--:|
|[Calibration Bottleneck: Over-compressed Representations are Less Calibratable](https://proceedings.mlr.press/v235/wang24cm.html)|PLP|ICML|2024|[![Star](https://img.shields.io/github/stars/dengbaowang/PLP.svg?style=social&label=Star)](https://github.com/dengbaowang/PLP)|[![Citations](badges/Calibration_Bottleneck__Over-compressed_Representations_are_Less_Calibratable.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Calibration+Bottleneck%3A+Over-compressed+Representations+are+Less+Calibratable&btnG=)|
|[When Does Label Smoothing Help?](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html?ref=gojiberries.io)||Label Smoothing|NeurIPS|2019|None|[![Citations](badges/When_Does_Label_Smoothing_Help_.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=When+does+label+smoothing+help%3F&btnG=)|

## Post-hoc Calibration
### Parametric Method
| Paper | Name| Source | Year| Code| Citations |
|-------|-------|-------|-------|-------|:--:|
|[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)|Temperature Scaling|ICML|2017|None|[![Citations](badges/On_Calibration_of_Modern_Neural_Networks.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=On+Calibration+of+Modern+Neural+Networks&btnG=)|
|[Intra Order-Preserving Functions for Calibration of Multi-Class Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2020/hash/9bc99c590be3511b8d53741684ef574c-Abstract.html)|Intra Order-Preserving Calibration|NeurIPS|2020|None|[![Citations](badges/Intra_Order-Preserving_Functions_for_Calibration_of_Multi-Class_Neural_Networks.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Intra+Order-Preserving+Functions+for+Calibration+of+Multi-Class+Neural+Networks&btnG=)|
|[Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)|Dirichlet Calibration|NeurIPS|2019|[![Star](https://img.shields.io/github/stars/dirichletcal/dirichletcal.github.io.svg?style=social&label=Star)](https://github.com/dirichletcal/dirichletcal.github.io)|[![Citations](badges/Beyond_temperature_scaling__Obtaining_well-calibrated_multi-class_probabilities_with_dirichlet_calibration.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Beyond+temperature+scaling%3A+Obtaining+well-calibrated+multi-class+probabilities+with+dirichlet+calibration&btnG=)|
|[Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classiﬁers](https://proceedings.mlr.press/v54/kull17a.html)|Beta Calibration|AISTATS|2017|[![Star](https://img.shields.io/github/stars/betacal/betacal.github.io.svg?style=social&label=Star)](https://github.com/betacal/betacal.github.io)|[![Citations](badges/Beta_calibration__a_well-founded_and_easily_implemented_improvement_on_logistic_calibration_for_binary_classi_ers.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Beta+calibration%3A+a+well-founded+and+easily+implemented+improvement+on+logistic+calibration+for+binary+classifiers&btnG=)|
|[Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods](https://www.researchgate.net/profile/John-Platt-2/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000/Probabilistic-Outputs-for-Support-Vector-Machines-and-Comparisons-to-Regularized-Likelihood-Methods.pdf)|Platt Scaling|Advances in Large Margin Classifiers|1999|None|[![Citations](badges/Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Probabilistic+Outputs+for+Support+Vector+Machines+and+Comparisons+to+Regularized+Likelihood+Methods&btnG=)|

### Non-parametric Method
| Paper | Name| Source | Year| Code| Citations |
|-------|-------|-------|-------|-------|:--:|
|[Obtaining Well Calibrated Probabilities Using Bayesian Binning](https://ojs.aaai.org/index.php/AAAI/article/view/9602)|Bayesian Binning|AAAI|2015|None|[![Citations](badges/Obtaining_Well_Calibrated_Probabilities_Using_Bayesian_Binning.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Obtaining+Well+Calibrated+Probabilities+Using+Bayesian+Binning&btnG=)|
|[Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers](https://dl.acm.org/doi/abs/10.5555/645530.655658)|Histogram Binning|ICML|2001|None|[![Citations](badges/Obtaining_calibrated_probability_estimates_from_decision_trees_and_naive_bayesian_classifiers.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Obtaining+calibrated+probability+estimates+from+decision+trees+and+naive+bayesian+classifiers&btnG=)|
|[Transforming classifier scores into accurate multiclass probability estimates](https://dl.acm.org/doi/abs/10.1145/775047.775151)|Isotonic Regression|KDD|2002|None|[![Citations](badges/Transforming_classifier_scores_into_accurate_multiclass_probability_estimates.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Transforming+classifier+scores+into+accurate+multiclass+probability+estimates&btnG=)|

### Mixed Method
| Paper | Name| Source | Year| Code| Citations |
|-------|-------|-------|-------|-------|:--:|
|[Mix-n-Match : Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning](https://proceedings.mlr.press/v119/zhang20k.html)|Mix-n-Match|ICML|2020|[![Star](https://img.shields.io/github/stars/zhang64-llnl/Mix-n-Match-Calibration.svg?style=social&label=Star)](https://github.com/zhang64-llnl/Mix-n-Match-Calibration)|[![Citations](badges/Mix-n-Match___Ensemble_and_Compositional_Methods_for_Uncertainty_Calibration_in_Deep_Learning.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Mix-n-Match+%3A+Ensemble+and+Compositional+Methods+for+Uncertainty+Calibration+in+Deep+Learning&btnG=)|
|[Verified Uncertainty Calibration](https://proceedings.neurips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)|Scaling-binning|NeurIPS|2020|[![Star](https://img.shields.io/github/stars/p-lambda/verified_calibration.svg?style=social&label=Star)](https://github.com/p-lambda/verified_calibration)|[![Citations](badges/Verified_Uncertainty_Calibration.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Verified+Uncertainty+Calibration&btnG=)|
|[Calibrating User Response Predictions in Online Advertising](https://link.springer.com/chapter/10.1007/978-3-030-67667-4_13)|SIR|ECML-PKDD|2020|None|[![Citations](badges/Calibrating_User_Response_Predictions_in_Online_Advertising.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Calibrating+User+Response+Predictions+in+Online+Advertising&btnG=)|
|[MBCT: Tree-Based Feature-Aware Binning for Individual Uncertainty Calibration](https://dl.acm.org/doi/abs/10.1145/3485447.3512096)|MBCT|WWW|2022|[![Star](https://img.shields.io/github/stars/huangsg1/Tree-Based-Feature-Aware-Binning-for-Individual-Uncertainty-Calibration.svg?style=social&label=Star)](https://github.com/huangsg1/Tree-Based-Feature-Aware-Binning-for-Individual-Uncertainty-Calibration)|[![Citations](badges/MBCT__Tree-Based_Feature-Aware_Binning_for_Individual_Uncertainty_Calibration.svg)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=MBCT%3A+Tree-Based+Feature-Aware+Binning+for+Individual+Uncertainty+Calibration&btnG=)|

# Calibration under Distribution Shift


# Calibration under Distribution Shift

## Calibration under Label Shift
| Paper | Source | Year| Code| Citations | 
|-------|-------|-------|-------|:--:|

## Calibration under Covariate Shift
| Paper | Source | Year| Code| Citations | 
|-------|-------|-------|-------|:--:|

## Calibration under Joint Shift
| Paper | Source | Year| Code| Citations | 
|-------|-------|-------|-------|:--:|

# How to Update Citations
Run the code:
```python
    Python Fetch_citation.py
