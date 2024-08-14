# SH_MC Methylation Classifier for Brain Tumors

SH_MC a novel methylation classification tool for brain tumors, improving accuracy with advanced techniques like SMOTE and OpenMax. 
It offers better performance on unknown or noisy data, a robust evaluation strategy, and calibrated score ranges. This tool enhances understanding and assessment of methylation profiling outcomes, 
contributing significant insights into brain tumor classification.
<img src="https://github.com/jaeminjj/OS_MC/blob/main/Figures/workflow.png" alt="workflow" width="500"/>
# Usage
After filtering and preprocessing data, we recommend to use GPU with Methylation Classifier.

Download two fine-tuned models for making Embedding vectors.
```python
https://huggingface.co/jaeminjj/EpicPred/tree/main
```
Please put two models in model folder

EpicPred environment requirements
```python
torch==2.3.1
transformers==4.42.3
sklearn==1.5.0
pandas==1.4.4
numpy==1.26.4
tensorflow==2.16.2
