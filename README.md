# HSimGCDA
HSimGCDA: A Novel Higher-order Similarity Measure Graph Convolutional Network for Predicting CircRNA-Disease Associations

## Abstract
Circular RNAs (circRNAs) have gained recognition for their critical roles in various biological processes and 
are increasingly viewed as promising candidates for use in disease diagnosis and treatment strategies for complex human diseases.
Despite  advances in computational methods for predicting circRNA-disease associations (CDAs), 
existing models often fail to effectively capture   both similarity metrics and higher-order mixed neighbourhood information simultaneously in bio-graph networks, leading to suboptimal prediction performance.
In this study, we present HSimGCDA, an enhanced graph convolutional network that leverages higher-order similarity measures to improve the prediction of CDAs. 
Specifically, our approach first constructs  multi-similarity graph networks that  integrate diverse data features of circRNAs and diseases.
We then implement a higher-order similarity strategy that  effectively aggregates the higher-order similarity information between nodes and their remote neighbors, thus enriching the topological information of the nodes and revealing potential features brought about by different neighboring nodes.
Ultimately, a multilayer perceptron algorithm is utilized to accurately infer likely  CDAs.
The experimental results indicated that HSimGCDA demonstrated excellent competitiveness on the benchmark CircR2Disease dataset, achieving a prediction accuracy of 95.93\% and an AUC score of 98.74\%. 
Additionally, the  ablation experiments  further validated the effectiveness of the improved HSimGCDA model.
Furthermore, case studies on gastric and breast cancers demonstrated that HSimGCDA successfully identified previously unrecognized circRNAs associated with these diseases.
Collectively, these findings emphasize the potential of HSimGCDA as a practical tool for exploring CDA, thereby providing valuable insights into disease diagnosis and therapeutic development. 

