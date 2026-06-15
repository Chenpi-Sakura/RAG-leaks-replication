# RAG-leaks: difficulty-calibrated membership inference attacks on retrieval-augmented generation

**Guangshuo WANG, Jiajun HE, Hao LI, Min ZHANG & Dengguo FENG**

*Special Topic: Enabling Techniques and Cutting-Edge Applications of Foundation Models*
*SCIENCE CHINA Information Sciences, June 2025, Vol. 68, Iss. 6, 160102:1–160102:18*

---

## Abstract

Recently, retrieval-augmented generation (RAG) systems have attracted attention for addressing issues like hallucinations and reliance on outdated knowledge in large language models (LLMs). Privacy studies have revealed that RAG systems are vulnerable to membership leakage in determining whether a specific target sample is included in the RAG knowledge base. Existing membership inference attack (MIA) methods for RAGs primarily rely on similarity scores between the system’s responses and the true answers. These methods assume that a higher similarity score indicates the sample is more likely to have been used by the RAG system to enhance its response, suggesting it is a member of the knowledge base. However, this study uncovers an important insight: the similarity metric does not directly represent the membership status, instead measures the response difficulty of the sample. To address this, we propose a novel membership inference attack for RAG systems, called difficulty-calibrated membership inference attack (DC-MIA). It first classifies high-similarity samples as members, and then calibrates the membership scores of samples with comparable raw similarity scores using a likelihood ratio test. Experimental results demonstrate that our approach significantly improves the performance of membership inference attacks on RAG systems.

**Keywords**: large language models, retrieval-augmented generation, membership inference, privacy risk, difficulty calibration

---

## 1 Introduction

In recent years, large language models (LLMs), such as GPT-4 and LLaMA, have garnered significant attention for their exceptional generative capabilities... (omitted detailed introductory text for brevity, focusing on core concepts).

Retrieval-augmented generation (RAG) has emerged as a promising approach to address these limitations, enhancing LLM responses by incorporating high-quality, timely, and contextually relevant information from external knowledge bases. However, when applied to tasks involving sensitive information, LLM-generated responses in RAG systems often rely on external knowledge bases containing private information, raising significant privacy concerns.

Studies have increasingly highlighted the significant privacy risks faced by RAG systems, with membership inference attacks (MIAs) being a key threat. Traditional MIAs aim to determine whether a specific data sample was involved in the training of a model, classifying it as a “member” if present or a “non-member” otherwise.

To address the limitations in existing MIAs on RAG systems, we reformulate the MIA problem in RAG systems and define the concept of response difficulty formally. Then, we propose difficulty-calibrated membership inference attacks (DC-MIA). DC-MIA builds on two key observations:
1. Samples generating answers with exceptionally high similarity (close to 1.0) are more likely to be members.
2. Due to varying levels of difficulty across different samples, there is significant overlap in similarity scores between members and non-members (ranging from 0.5 to 0.9).

Leveraging these insights, we partition the sample space into two regions: a high-similarity region, primarily containing members, and a confusion region, where similarity scores of members and non-members overlap. Samples in the high-similarity region are directly classified as members, while those in the confusion region undergo a likelihood ratio test.

## 2 Related work
### 2.1 Retrieval augmented generation
A typical RAG system comprises three key components: an LLM $\mathcal{M}$, a retriever $\mathcal{R}$, and an external knowledge base $D$. For each generation, the retriever fetches the Top-k most relevant text records in response to the user's query $q$. These are provided along with $q$ to the LLM $\mathcal{M}$. 

### 2.2 Membership inference attack
MIAs are a common privacy threat. In RAGs, the adversary seeks to infer whether a sample is part of the non-parametric knowledge base. Existing works like Anderson et al. and Li et al. infer membership by calculating similarity or querying the LLM directly, but they often ignore response difficulty.

## 3 Problem statement
### 3.1 Membership inference attack on RAG
**Adversary’s objective**: Determine whether a specific target sample $x$ is included in the RAG knowledge base $D$.
**Attack model**: Uses a pre-defined threshold on a membership score (e.g., similarity $S_R(a')$) to predict membership: $\hat{b} = I(\mathcal{A}(x) > \tau)$.

### 3.2 Difficulty calibration on RAG
**Definition 2 (Response difficulty)**: The response difficulty of a text refers to how difficult it is for the RAG system to comprehend the text. It consists of two parts:
$\text{DIF}_{\text{response}}(x, \mathcal{G}) = \text{DIF}_{\text{sample}}(x) + \text{DIF}_{\text{rag}}(x, \mathcal{G})$

- $\text{DIF}_{\text{sample}}$: Depends solely on the sample's intrinsic characteristics and the LLM's pre-trained knowledge (sample diversity).
- $\text{DIF}_{\text{rag}}$: Influenced by whether $x$ exists in the RAG knowledge base (membership status).

### 3.3 Threat model
Black-box scenario: The adversary only has black-box access to the target RAG system (can submit query $q$ and observe answer $a'$). They know the target RAG's architecture/parameters (can reconstruct local replicas) and the data distribution, plus hold a small auxiliary dataset of known members.

## 4 Attack methodology
### 4.1 Intuition
1. Isolate the high similarity score region and classify those samples as members.
2. Use reference RAGs and incorporate sample difficulty to design a calibrated membership score, improving accuracy in the confusion region.

### 4.2 Overview of DC-MIA
Performs a two-phase attack on a target text:
1. **Phase 1: High-similarity regions recognition**: Determine a threshold $\tau_1$ using an auxiliary dataset. If a target sample's similarity $> \tau_1$, it's classified as a member.
2. **Phase 2: Confusion regions identification**: Construct reference inRAGs and outRAGs. Calculate similarity in these local reference models, fit distributions, and compute a likelihood ratio test to determine if the actual similarity score is closer to the inRAG or outRAG distribution.

### 4.4 Phase 2: confusion regions identification
Formulated as a hypothesis test using a likelihood ratio:
$\Lambda(x) \approx \frac{p(S_R(a')|\tilde{Q}_{\text{in}}(x))}{p(S_R(a')|\tilde{Q}_{\text{out}}(x))}$

If $\Lambda(x) > \tau_2$, the sample is a member.

## 5 Experimental setup
### 5.1 Datasets
- **HealthCareMagic**: Medical dialogues (~110,000 data points).
- **AgNews**: Text classification/news articles (~120,000 articles).
- **NaturalQuestions**: Open-domain QA.
Target RAG has 8000 members. Evaluated on 1000 members + 1000 non-members.

### 5.2 RAG systems
- **LLMs**: Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-V02, glm-4-9b-chat.
- **Retriever**: all-MiniLM-L6-v2 with FAISS.
- **Prompt**: "Please answer the question based on the provided context. Context: {context} Question: {user_prompt}"

## 6 Experimental results
- **Attack performance**: DC-MIA consistently outperforms the baseline (S-MIA) in terms of AUC and TPR at 1% FPR across different datasets and LLMs.
- **Ablation study**: 
  - *Phase-2-only*: Shows that the likelihood ratio test is the core contributor to success.
  - *Number of reference RAGs*: Performance peaks around 16 reference RAGs.
  - *Number of retrieved items (Top-k)*: Retrieving fewer items (e.g., $k=1$) increases privacy risk, while higher $k$ slightly decreases attack success but maintains overall vulnerability.
  - *Retriever type*: DC-MIA is robust across different retrievers (BGE-en, BM25, and ideal retrieval).

## 7 Discussion
- Membership inference shouldn't rely purely on cosine similarity because similarity measures response difficulty, which is inherently affected by sample diversity.
- Other metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum) can also be calibrated using DC-MIA and sometimes yield even better attack performance depending on the RAG setup.

## 8 Conclusion and future work
DC-MIA specifically addresses the impact of sample difficulty on MIA performance. It outperforms baselines by a significant margin. Future work should investigate privacy-preserving defense mechanisms for RAG systems to ensure their security and confidentiality.
