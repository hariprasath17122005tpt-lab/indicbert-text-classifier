# Viva Voce Questions & Answers: IndicBERT Multilingual Text Classifier

This guide contains 32 core questions and concise, technically rigorous answers tailored for B.Tech Computer Science (AI/ML) project viva examinations.

---

### Q1: What is Multilingual AI?
**Answer**: Multilingual AI refers to machine learning systems engineered to understand, process, generate, and translate text across multiple natural languages using unified models and shared semantic representations, rather than maintaining independent models for every language.

### Q2: What is Natural Language Processing (NLP)?
**Answer**: NLP is a subfield of artificial intelligence that gives computers the ability to understand, interpret, and manipulate human language by combining computational linguistics, statistical modeling, and deep learning.

### Q3: What are Indic languages and what are their major language families?
**Answer**: Indic languages are the languages indigenous to the Indian subcontinent. They predominantly belong to two primary families:
1. **Indo-Aryan** (e.g., Hindi, Bengali, Punjabi, Gujarati, Marathi)
2. **Dravidian** (e.g., Tamil, Telugu, Malayalam, Kannada)

### Q4: What defines a "low-resource" language in NLP?
**Answer**: A low-resource language is one that lacks sufficient computational resources, such as large digital corpora, annotated training datasets, standardized lexicons, or high-performance language technologies, making standard supervised deep learning difficult.

### Q5: What is IndicBERT?
**Answer**: IndicBERT is a multilingual language model developed by AI4Bharat and IIT Madras, pretrained on massive corpora across 12–23 Indian languages. It shares vocabulary and embeddings across Indic scripts, enabling cross-lingual transfer learning.

### Q6: What is BERT (Bidirectional Encoder Representations from Transformers)?
**Answer**: BERT is a transformer-based language representation model introduced by Google (Devlin et al., 2018). Unlike directional models that read text left-to-right or right-to-left, BERT’s self-attention mechanism processes tokens in both directions simultaneously.

### Q7: What is the architectural difference between original IndicBERT and IndicBERT v2?
**Answer**: Original IndicBERT (`ai4bharat/indic-bert`) utilized an **ALBERT** architecture with parameter sharing across layers (~33M parameters). IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`) adopts a standard **BERT-base** architecture (~278M parameters) supporting 23 Indian languages with enhanced vocabulary coverage and representation power.

### Q8: What is Transfer Learning?
**Answer**: Transfer learning is a machine learning paradigm where a model pretrained on a large-scale, general task (e.g., Masked Language Modeling on billions of web tokens) is adapted or fine-tuned on a specific downstream task (e.g., sentiment analysis on a few thousand samples).

### Q9: What is Fine-Tuning?
**Answer**: Fine-tuning is the process of taking a pretrained model, attaching a task-specific head (e.g., a linear classification layer), and updating its parameters using a smaller labeled dataset with a low learning rate.

### Q10: What is Tokenization in NLP?
**Answer**: Tokenization is the step of segmenting raw text strings into smaller discrete units called tokens (words, subwords, or characters) that can be mapped to numerical IDs in the model's vocabulary.

### Q11: Why do transformer models use Subword Tokenization rather than word-level tokenization?
**Answer**: Word-level tokenization cannot handle out-of-vocabulary (OOV) words and explodes in vocabulary size for agglutinative languages like Tamil or Malayalam. Subword tokenization (e.g., WordPiece, BPE) splits unfamiliar words into known subword fragments (e.g., "நன்றாக" $\rightarrow$ "நன்", "##றாக"), ensuring 100% vocabulary coverage.

### Q12: What are Token IDs and Attention Masks?
**Answer**:
- **Token IDs**: Numerical integer representations corresponding to each subword token in the model's vocabulary lookup table.
- **Attention Mask**: A binary tensor (1s for real tokens, 0s for padding tokens) instructing the transformer self-attention mechanism to ignore padding.

### Q13: What is a Sequence Classification Head?
**Answer**: A classification head is a lightweight feed-forward neural network layer (typically a Dropout layer and a Linear transformation $768 \rightarrow \text{num\_classes}$) placed on top of the transformer's pooled representation (the `[CLS]` token embedding) to map hidden states to class logits.

### Q14: Why is using a Pretrained Model superior to training a classifier from scratch?
**Answer**: Training from scratch on small datasets (a few thousand samples) results in severe overfitting and poor generalization. Pretrained models possess prior syntactic and semantic knowledge of language structure, requiring far fewer labeled samples to reach high accuracy.

### Q15: What is Sentiment Analysis?
**Answer**: Sentiment analysis is an NLP task that identifies, extracts, and quantifies the subjective affective state or emotional polarity (e.g., Positive, Negative, Neutral) expressed in a text.

### Q16: What is Data Leakage and how did you prevent it?
**Answer**: Data leakage occurs when information from outside the training dataset (such as test or validation samples) contaminates the training process, producing unrealistically optimistic scores. We prevented leakage by performing stratified train/validation/test splitting *before* training, deduplicating records, and evaluating *only* on the isolated test set.

### Q17: What is Stratified Splitting?
**Answer**: Stratified splitting ensures that each split (train, validation, test) contains approximately the exact same proportion of class labels and language representation as the complete dataset, preventing representation skew.

### Q18: What is Overfitting and how do you mitigate it?
**Answer**: Overfitting occurs when a model memorizes the training data's noise rather than learning generalizable patterns, resulting in low training error but high test error. We mitigate it using validation-based early stopping, dropout ($p=0.1$), weight decay ($0.01$), and base encoder parameter freezing.

### Q19: What is Accuracy and when can it be misleading?
**Answer**: Accuracy is the ratio of correct predictions to total predictions: $\frac{TP + TN}{TP + TN + FP + FN}$. It is misleading when classes are imbalanced; for example, if 95% of data is Positive, a model predicting Positive 100% of the time achieves 95% accuracy despite learning nothing about Negative sentiment.

### Q20: What are Precision and Recall?
**Answer**:
- **Precision**: The proportion of predicted positive instances that were actually positive: $\frac{TP}{TP + FP}$.
- **Recall**: The proportion of actual positive instances that were correctly identified: $\frac{TP}{TP + FN}$.

### Q21: What is the F1-Score?
**Answer**: The F1-score is the harmonic mean of precision and recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. It provides a balanced metric when an even balance between precision and recall is required.

### Q22: What is Macro F1-Score and why is it our primary evaluation metric?
**Answer**: Macro F1 calculates the F1-score independently for each class and takes the unweighted arithmetic mean. It treats all classes equally regardless of sample count, making it the gold standard for low-resource and imbalanced NLP tasks.

### Q23: What is a Confusion Matrix?
**Answer**: A confusion matrix is an $N \times N$ table that visualizes classifier performance by cross-tabulating actual ground-truth classes (rows) against predicted classes (columns), highlighting True Positives, False Positives, and False Negatives.

### Q24: What was the purpose of the Low-Resource Learning Experiment?
**Answer**: The low-resource experiment trains the model on varying fractions of the training corpus (10%, 25%, 50%, 100%) and tests on an identical held-out test set. It empirically demonstrates the data efficiency of IndicBERT transfer learning when annotated data is scarce.

### Q25: Why must native Indic scripts be preserved instead of transliterating to English?
**Answer**: Indian scripts are phonetically structured (alphasyllabaries / abugidas). Transliteration into the Latin alphabet (Romanization) introduces ambiguities, inconsistent phonetic spellings, and information loss. Preserving native Unicode ensures morphological features are accurately modeled.

### Q26: What is Hugging Face?
**Answer**: Hugging Face is an open-source AI platform and ecosystem providing standardized libraries (`transformers`, `datasets`, `accelerate`) and a model repository (Hub) for sharing and deploying machine learning models.

### Q27: What is Streamlit?
**Answer**: Streamlit is an open-source Python framework designed for rapidly creating interactive, data-driven web applications for machine learning and data science projects without requiring frontend web technologies (HTML/CSS/JS).

### Q28: Why did you use local CSV storage instead of a relational database like MySQL?
**Answer**: Local CSV storage with UTF-8 encoding satisfies the project's zero-external-dependency requirement. It ensures immediate portability across student laptops and Hugging Face Spaces without requiring database daemon configuration, connection strings, or credentials.

### Q29: What is the Softmax function?
**Answer**: Softmax converts a vector of arbitrary real-valued logits into a normalized probability distribution where each value is bounded in $[0, 1]$ and all values sum to $1$: $P(y=i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$.

### Q30: What is the AdamW Optimizer?
**Answer**: AdamW is an optimization algorithm that decouples weight decay from the gradient updates of Adam. This fixes the incorrect weight decay regularization in standard L2-regularized Adam, leading to superior generalization in transformer fine-tuning.

### Q31: What are the main limitations of this project?
**Answer**:
1. **Script Ambiguity**: Romanized or code-mixed Indian text (e.g., Hinglish or Tanglish) requires specialized transliteration models.
2. **Contextual Nuances**: Sarcasm, irony, and figurative language remain challenging for lightweight sentiment heads.
3. **Hardware Constraints**: Full 278M parameter fine-tuning on CPU requires substantial memory, necessitating layer freezing on consumer hardware.

### Q32: How can this project be enhanced in the future?
**Answer**:
1. Adding support for multi-class emotion detection (Joy, Sadness, Anger, Neutral).
2. Incorporating parameter-efficient fine-tuning (PEFT) techniques like LoRA (Low-Rank Adaptation).
3. Extending to code-mixed Dravidian text (DravidianCodeMix datasets).
4. Integrating model quantization (8-bit / 4-bit) with ONNX Runtime for mobile deployment.
