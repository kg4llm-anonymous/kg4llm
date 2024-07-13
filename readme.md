### Files

* api_tree.py: This file provides a package for constructing a TrieTree to speed up the process of retrieving APIs from the API knowledge base during knowledge-guided beam search.
* bleu.py: This file provides a package for calculating the BLEU metrics.
* checker.py: This file provides a package for checking whether the recommended APIs contain untrustworthy issues (UTIs) containing: (1) Including fictitious APIs (UTI 1). (2) Including APIs with unsatisfied calling conditions (UTI 2). (3) Failures to conform to interface parameter types (UTI 3).
* guider.py: This file provides a package for realizing knowledge-guided beam search.
* interactive.py: The entry for interacting with the API recommendation system.
* model.py: This file defines the LLM structure and provides finetuning and generating interfaces.
* run.py: The entry for training and testing.
* syn_bleu: This file provides a package for calculating the TR and TBLEU metrics.
* KG4LLM/ChatGPT/DeepSeek.cases.json: The total 60 real-world cases and their manually analyzed results.

### Data and Environment Preparation

1. Dataset and API knowledge base
   Please download the dataset and knowledge base from "https://drive.google.com/file/d/1zurxfo59CH4Bhu8NJdWiP9XZs68983k6/view?usp=drive_link". Then, put them in the same directory with the code.
2. Develop Environment

- python 3.10.0

- pytorch 2.0.1

- transformers 4.30.0

### Finetuning

To finetune LLM using Knowledge-guided Data Augmentation, you can directly execute the following instruction:

```python run.py --do_train --do_eval```

The pre-trained LLM parameters will be downloaded automatically.

When finished finetuning, you can find the LLM parameter file at "./model/checkpoint-best-bleu/pytorch_model.bin"

### Calculate metrics on Testing Set (RQ1, RQ2 and RQ3)

Ensure you have finetuned LLM and that you can find the LLM parameter file at "./model/checkpoint-best-bleu/pytorch_model.bin."

To test LLM using Knowledge-guided Beam Search on the testing set, you can directly execute the following instruction:

```python run.py --do_test --load_model_path ./model/checkpoint-best-bleu/pytorch_model.bin```

By executing the instruction, the metrics (e.g., BLEU@1/5/10, TR1@1/5/10, TR2@1/5/10, TR3@1/5/10, TR@1/5/10, TBLEU@1/5/10) mentioned in RQ1/2/3 will be output in the terminal. You can also find these metrics at "./model/output.log".

Besides, you can also find the Top-10 API recommendation results for each query in the testing set  "./model/test_rate=0.1_top10.output".

### Interactive Q&A for Case Analysis (RQ4)

Ensure you have finetuned LLM and that you can find the LLM parameter file at "./model/checkpoint-best-bleu/pytorch_model.bin".

To interactive with LLM-based API recommender for case analysis (RQ4), you can directly execute the following instruction:

```python interactive.py --load_model_path ./model/checkpoint-best-bleu/pytorch_model.bin```

By executing the instruction, the terminal will prompt you to enter functional description and interface parameter types. After enter the required information, top-10 recommendation results will be output. A demo is as follows:

```
# input
Please input your functional description: create a camel endpoint uri based on the component and service name
Please input your interface parameter types (add a space between each parameter type): java.lang.StringBuilder javax.xml.namespace.QName java.lang.CharSequence

# output
1: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.Object) javax.xml.namespace.QName.toString() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

2: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.Object) javax.xml.namespace.QName.getNamespaceURI() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

3: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.Object) javax.xml.namespace.QName.getLocalPart() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

4: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.String) javax.xml.namespace.QName.toString() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

5: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.CharSequence) javax.xml.namespace.QName.toString() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

6: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.String) javax.xml.namespace.QName.getLocalPart() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

7: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(boolean) javax.xml.namespace.QName.getLocalPart() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

8: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.String) javax.xml.namespace.QName.getNamespaceURI() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

9: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.CharSequence,int,int) javax.xml.namespace.QName.getLocalPart() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()

10: java.lang.StringBuilder.StringBuilder(java.lang.CharSequence) java.lang.StringBuilder.append(java.lang.CharSequence) javax.xml.namespace.QName.getLocalPart() java.lang.StringBuilder.append(java.lang.String) java.lang.StringBuilder.toString()
```










