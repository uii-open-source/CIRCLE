## Introduction

This repository is the official repository of "A Comprehensive Foundation Model for Broad-Spectrum Thoracic Abnormality Detection in Non-Contrast CT Scans".
<p align="center">
  <img src="figures/circle.png" width="100%">
</p>


## Training

For details on the training of CIRCLE models, please navigate to [train](train).


## The CIRCLE-ZS2K Dataset

_**Access**_

We randomly selected 2,000 cases from our internal validation set for open release, including both images and their corresponding reports (CIRCLE-ZS2K dataset), which can be publicly accessed via [Hugging Face repository](https://huggingface.co/datasets/wangxiaoyuwangdayu/CIRCLE). At the present stage, ​​it is mandatory to​​ contact the CIRCLE collaborators ​​prior to​​ any academic publication that utilizes this dataset. The use of CIRCLE-ZS2K dataset for any commercial purposes ​​is strictly prohibited​​.

_**Copyright, usage, and ethical considerations**_

The CIRCLE-ZS2K dataset is an example dataset associated with a related publication. The corresponding manuscript is currently under submission, and we plan to release a larger-scale version of the dataset—subject to obtaining further ethical approvals and administrative permissions—if feasible. The CIRCLE collaborators consider the protection of patient interests and privacy to be of utmost importance. Accordingly, all personally identifiable information contained in the dataset has been anonymized.

_**Structure**_

The overall structure of the CIRCLE-ZS2K dataset is organized as follows:

| Directory               | Content |
|-------------------------|---------|
| CIRCLE-ZS2K/image/...   | 2,000 sample CT scans in each subdirectory, numbered from 0001 to 2000. Each case includes a non-contrast CT image stored in the NIfTI format (CT.nii.gz). |
| CIRCLE/report/...       | The file report.csv comprises three columns: <br> **image_name**: identifies each case by its corresponding number (1–2000). <br> **finding**: describes the imaging observations in Chinese, including anatomical and pathological findings from the non-contrast CT scan, as reviewed by senior radiologists. <br> **impression**: provides the diagnostic conclusion in Chinese, also validated by senior radiologists. |

_**Example (Index 0002)**_

​​**Finding​​**

左上肺舌段见类圆形肿块影，大小约65*56mm，增强后边缘强化为主，中心坏死无明显强化，病灶周围见斑片模糊影及条片状实变影；两肺少许斑点条索影，左下肺少许实变影，所见各支气管腔通畅，纵隔见轻度肿大淋巴结，左侧胸腔内少量积液。

(English translation) A round-like mass measuring approximately 65 × 56 mm is observed in the lingular segment of the left upper lobe. Post-contrast imaging shows predominantly peripheral enhancement with central necrosis demonstrating no significant enhancement. Patchy opacities and streaky consolidation are seen around the lesion. A few spotted and streaky shadows are noted in both lungs, along with areas of consolidation in the left lower lobe. The bronchial airways are patent. Mildly enlarged lymph nodes are present in the mediastinum, accompanied by a small amount of pleural effusion on the left side.

​​**Impression​​**

左上肺MT伴阻塞性炎症及不张，左侧少量胸腔积液；两肺少许慢性炎症陈旧灶，左下肺少许节段性不张。

(English translation) Malignant tumor in the left upper lobe with obstructive inflammation and atelectasis; left-sided small pleural effusion; few chronic inflammatory and fibrotic changes in both lungs; segmental atelectasis in the left lower lobe.


