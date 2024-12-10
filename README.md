

# Query-aware Multi-scale Proposal Network for Weakly Supervised Temporal Sentence Grounding in Videos  
This repository contains the code and resources for the paper:  
**"Query-aware Multi-scale Proposal Network for Weakly Supervised Temporal Sentence Grounding in Videos"** (Knowledge-Based Systems, 2024).  

![Model Overview](imgs/method.png)  

## Table of Contents  
- [Overview](#overview)  
- [Data Preparation](#data-preparation)  
- [Environment Setup](#environment-setup)  
- [Running Experiments](#running-experiments)  
- [Citation](#citation)  

---

## Overview  
This project introduces a novel Query-aware Multi-scale Proposal Network, designed for weakly supervised temporal sentence grounding in videos. The model effectively leverages query-aware multi-scale proposals to improve temporal grounding accuracy under weak supervision.  

For a visual overview of the model, see `imgs/method.png`.  

---

## Data Preparation  
### Charades-STA and ActivityNet Captions  
- **Charades-STA Features**: Use the features provided by [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding). See the [Charades-STA repository](https://github.com/JonghwanMun/LGI4temporalgrounding) for details.  
- **ActivityNet Captions Features**: Download hdf5 features from [ActivityNet Captions](http://activity-net.org/challenges/2016/download.html).  

Download the features and place them in the corresponding directories under `data/`.  
Refer to [CPL](https://github.com/minghangz/cpl) for additional details on feature preparation.  

### TACoS and EgoVLP  
- **TACoS Features**: Contact the first author via email to obtain the feature files.  
- **EgoVLP Features**: Similarly, contact the first author for access to these features.  

### Important Note  
Ensure you modify the `config` JSON files to set the correct `feature_path` for each dataset.  

---

## Environment Setup  
Follow the environment setup instructions in:  
**"Counterfactual Cross-modality Reasoning for Weakly Supervised Video Moment Localization"** (ACM MM 2023).  

This will guide you in preparing the required dependencies and configurations.  

---

## Running Experiments  
To conduct experiments on the four datasets (Charades-STA, ActivityNet Captions, TACoS, and EgoVLP), execute the corresponding bash scripts provided in the repository.  
Before running the test, please create a new folder for the error path to store the results.
---

## Citation  
If you find this work helpful in your research, please consider citing:  

```bibtex  
@article{zhou2024query,  
  title={Query-aware multi-scale proposal network for weakly supervised temporal sentence grounding in videos},  
  author={Zhou, Mingyao and Chen, Wenjing and Sun, Hao and Xie, Wei and Dong, Ming and Lu, Xiaoqiang},  
  journal={Knowledge-Based Systems},  
  volume={304},  
  pages={112592},  
  year={2024},  
  publisher={Elsevier}  
}  
```  

---  
For additional inquiries or support, feel free to contact the first author via email.  
