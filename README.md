# CULtural Text Understanding and Relevance Index(CULTURIX)
Here's a README template for your GitHub project based on the provided details:

---

# German Language Model Evaluation for Hallucination Detection

This repository contains the code and resources for our paper "A Novel Approach for Evaluating Hallucinations in German Language Large Language Models (LLMs)," submitted to COLM 2024. Our research introduces a groundbreaking method tailored for assessing the accuracy and cultural relevance of German LLMs, addressing the unique challenges posed by the German language's complex grammatical structure and cultural nuances.

## Overview

Our study focuses on developing an advanced LLM specifically for the German language, leveraging the DistilBERT model for its efficiency and adaptability in language tasks. We aim to significantly reduce the occurrence of hallucinations in LLM outputs, ensuring the generated text remains both factually accurate and culturally relevant.

## Repository Structure

- `README.md` - This document providing an overview and instructions for the repository.
- `requirements.txt` - The Python package dependencies required to run the code.
- `data/` - Directory containing the dataset used for training and evaluation (note: due to privacy and licensing, actual data might not be included).
- `src/` - Contains the Python scripts for model training, evaluation, and utility functions.
  - `model_training.py` - Script for training the DistilBERT model on our dataset.
  - `data_preparation.py` - Utility functions for data loading and preprocessing.
- `notebooks/` - Jupyter notebooks illustrating the model's usage and evaluation results.
- `results/` - Directory where training results and evaluation metrics are saved.

## Getting Started

## Dataset

Please download the dataset from hugging face! 

https://huggingface.co/datasets/srreyS/CULTIRX_Identification

### Prerequisites

Ensure you have Python 3.6+ installed on your system. You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

### Training the Model

To start training the model with the provided dataset, run:

```bash
python src/model_training.py
```

This script will preprocess the data, train the model, and save the trained model along with evaluation metrics in the `results/` directory.

### Evaluation

The training script automatically evaluates the model on a test set and prints out accuracy, precision, recall, and F1-score. Further analysis can be performed using the Jupyter notebooks in the `notebooks/` directory.

## Contributing

We welcome contributions to improve the model and its evaluation. Please feel free to submit issues and pull requests.

## License
(BLANK)

## Citation

```bibtex
CITATION
```


**Refereneces:** 

ambrosfitz. American yawp full dataset, 2023. Hugging Face. Ashishkr. 

Query wellformedness score. https://huggingface.co/Ashishkr/query_wellformedness_score. Hugging Face Model Hub.

Branden Chan, Timo M  ̈oller, Malte Pietsch, and Tanay Soni. bert-base-german-cased, 2019. Hugging Face, Published: Jun 14th, 2019.

Qinyuan Cheng, Tianxiang Sun, Wenwei Zhang, Siyin Wang, Xiangyang Liu, Mozhi Zhang, Junliang He, Mianqiu Huang, Zhangyue Yin, Kai Chen, et al. Evaluating hallucinations in chinese large language models, 2023. arXiv preprint arXiv:2310.03368.

Dhruvil Dave and Aneri Dalwadi. Wikibooks dataset, 2021. 

Kaggle, October 22, 2021. The Devastator. 

Folk tales, 2023. Kaggle. Edenbd. 

Children stories text corpus, 2023. Kaggle. Dafydd Gibbon. 

Language in the computer age: A note on current activities in computational linguistics in germany, 1987. The Finite String Newsletter, 

Language in the Computer Age Section, Computational Linguistics, Volume 13, Numbers 3-4, July-December 1987, pp. 364-365.

Nuno M. Guerreiro, Duarte Alves, Jonas Waldendorf, Barry Haddow, Alexandra Birch, Pierre Colombo, and Andr  ́e F. T. Martins. Hallucinations in large multilingual translation models, 2023.

Dell Research Harvard. Americanstories (revision 3484aca), 2023. Hugging Face.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. True: Re-evaluating factual consistency evaluation, 2022. arXiv preprint arXiv:2204.04991, primaryClass: cs.CL.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. Not specified, 2023.

Siqing Huo, Negar Arabzadeh, and Charles L. A. Clarke. Retrieving supporting evidence for llms generated answers. 2023.

Leonard B. Jung, Jonas A. Gudera, Tim L. T. Wiegand, Simeon Allmendinger, Konstantinos Dimitriadis, and Inga K. Koerte. Chatgpt passes german state examination in medicine with picture questions omitted. Dtsch Arztebl Int, 120(21-22):373–374, 2023. PMC10413971, May 2023.

D. Khurana, A. Koli, K. Khatter, and S. Singh. Natural language processing: state of the art, current trends and challenges. Multimedia Tools and Applications, 82:3713–3744, 2023.

Deren Lei, Yaxi Li, Mengya Hu, Mingyu Wang, Vincent Yun, Emily Ching, and Eslam Kamal. Chain of natural language inference for reducing large language model ungrounded hallucinations. 2023.

Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 6449–6464, Singapore, 2023. Association for Computational Linguistics.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74–81, Barcelona, Spain, 2004. Association for Computational Linguistics.

Potsawee Manakul, Adian Liusie, and Mark Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9004–9017, Singapore, 2023. Association for Computational Linguistics.

Alessia McGowan, Yunlai Gui, Matthew Dobbs, Sophia Shuster, Matthew Cotter, Alexandria Selloni, Marianne Goodman, Agrima Srivastava, Guillermo A. Cecchi, and Cheryl M. Corcoran. Chatgpt and bard exhibit spontaneous citation fabrication during psychiatry literature search. Psychiatry Research, 326:115334, 2023.

Saurabh Pahune and Manoj Chandrasekharan. Several categories of large language models (llms): A short survey, 2023. arXiv:2307.10188 [cs.CL], July 2023.

Bj  ̈orn Pl  ̈uster. Leolm: Igniting german-language llm research, 2023. LAION, 28 Sep, 2023. M. Popovi  ́c. Language-related issues for nmt and pbmt for english–german and english–serbian. Machine Translation, 32:237–253, 2018.

Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, S. M Towhidul Islam Tonmoy, Aman Chadha, Amit P. Sheth, and Amitava Das. The troubling emergence of hallucination in large language models – an extensive definition, quantification, and prescriptive remediations. In Not specified, 2023a.

Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation models. Arxiv, 2023b.

Jonas Roos, Adnan Kasapovic, Tom Jansen, and Robert Kaczmarczyk. Artificial intelligence in medical education: Comparative analysis of chatgpt, bing, and medical students in germany. JMIR Medical Education, 2023. 4.9.2023.

Tanushree Shenwai. Another large language model! meet igel: An instruction-tuned german llm family, 2023. Marktechpost, July 18, 2023.

TheITCrow. German bundestag speeches: Translated & summarized, 2023. Kaggle, December 2023.

S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Amitava Das. A comprehensive survey of hallucination mitigation techniques in large language models. 2024.

Stefanie Urchs, Veronika Thurner, Matthias Aßenmacher, Christian Heumann, and Stephanie Thiemichen. How prevalent is gender bias in chatgpt? – exploring german and english chatgpt responses, 2023. arXiv:2310.03031 [cs.CL], 21 Sep 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

Chin Man Yeung. Effects of inserting domain vocabulary and fine-tuning bert for german legal language, 2019.
