# IMDB Sentiment Analysis (ID3 + AdaBoost)

## Overview

This project implements sentiment classification on the IMDB dataset using:

* A custom ID3 decision tree
* A custom AdaBoost ensemble (with ID3)
* Scikit-learn AdaBoost for comparison

The task is binary classification:

* 0 = negative review
* 1 = positive review

---

## Dataset

IMDB dataset from TensorFlow:

* 25,000 training samples
* 25,000 test samples
* Only top 4000 words kept (`num_words=4000`)

---

## Preprocessing

1. Load IMDB dataset
2. Convert word indices back to text
3. Build vocabulary from training data
4. Convert each review into a **binary bag-of-words vector**:

   * 1 if word exists in review
   * 0 otherwise

Output shape:

```
(n_samples, vocabulary_size)
```

---

## ID3 Decision Tree

Custom implementation of ID3 using **information gain**.

### Structure

**Node**

* `checking_feature`
* `left_child`
* `right_child`
* `isLeaf`
* `category`

### Main methods

* `fit(X, y)` → builds tree
* `create_tree(...)` → recursive construction
* `calculate_ig(...)` → entropy / information gain
* `predict(X)` → classification

---

## AdaBoost (Custom ID3)

Ensemble of ID3 trees.

### Algorithm

For each estimator:

1. Sample data using weights
2. Train ID3 tree
3. Compute weighted error
4. Compute alpha:

```
alpha = 0.5 * log((1 - err) / err)
```

5. Update sample weights

Final prediction:

```
sign(sum(alpha * prediction))
```

---

## Evaluation

Metrics used:

* Precision
* Recall
* F1-score

Evaluated on:

* Positive class
* Negative class
* Macro average
* Micro average

---

## Learning Curves

Tracks:

* Precision
* Recall
* F1-score

Against training set size using a development split.

---

## Scikit-learn Comparison

Used:

* `AdaBoostClassifier`
* Decision stumps (`DecisionTreeClassifier(max_depth=1)`)

Compared against custom implementation.

---

## Known Issues

### 1. Slow training

Binary bag-of-words with ~4000 features is expensive with custom loops.

### 2. Kernel crashes

Caused by:

* Large dense feature matrix
* Inefficient ID3 implementation

### 3. AdaBoost label format

Current code uses `{0,1}` but AdaBoost is more stable with `{-1,1}`.

---

## Suggested Improvements

* Use sparse matrices instead of dense lists
* Replace manual vocabulary loop with `CountVectorizer`
* Limit tree depth (pruning)
* Convert labels to `{-1, 1}` for boosting
* Reduce vocabulary size

---

## Requirements

```bash
tensorflow
numpy
scikit-learn
matplotlib
tqdm
```

---

## Summary

This project demonstrates:

* Manual implementation of ID3
* Boosting with weak learners
* Comparison with standard ML library implementation
