# Fairness-Guided Few-Shot Prompting

This document describes the theory and mathematical formulation behind **T-fair-Prompting** and **G-fair-Prompting**, two strategies for selecting demonstrations in few-shot prompting using large language models.

---

## 1. Problem Setting

We consider a classification task with a labeled training set

$$
\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{N}
$$

where  
- $x_i$ is an input text  
- $y_i \in \mathcal{Y}$ is its label  
- $\mathcal{Y}$ is the set of all class labels

A **demonstration** is constructed using a fixed template

$$
\Gamma(x_i, y_i)
$$

For sentiment classification, this corresponds to
$$
review: x_i
$$
$$
label: y_i
$$

A **prompt** is formed by concatenating multiple demonstrations

$$
\rho = \Gamma(x_{i_1}, y_{i_1}) \oplus \Gamma(x_{i_2}, y_{i_2}) \oplus \cdots
$$

---

## 2. Content-Free Evaluation

To measure the intrinsic bias of a prompt, we evaluate it on a **content-free input**

$$
\eta \in \{\text{N/A}, \text{""}, \text{" "}\}
$$

The full input to the language model is

$$
\rho \oplus \eta
$$

Since $\eta$ contains no semantic information, an ideal prompt should induce a **uniform predictive distribution** over labels.

---

## 3. Predictive Distribution

Given a prompt $\rho$, the language model produces a probability distribution over labels

$$
\hat{p}(y \mid \rho \oplus \eta)
$$

In practice, this distribution is obtained by:
1. Taking the logits of the next generated token
2. Selecting logits corresponding to label tokens
3. Applying softmax over the label set $\mathcal{Y}$

---

## 4. Fairness as Predictive Entropy

The **fairness score** of a prompt is defined as the entropy of the predictive distribution

$$
\text{fair}(\rho)
=
- \sum_{y \in \mathcal{Y}}
\hat{p}(y \mid \rho \oplus \eta)
\log \hat{p}(y \mid \rho \oplus \eta)
$$

Interpretation:

- High entropy  
  ⇒ low predictive bias  
  ⇒ fair prompt  

- Low entropy  
  ⇒ strong label preference  
  ⇒ biased prompt  

This metric allows prompt evaluation **without a validation set**.

---

## 5. T-fair-Prompting (Top-k Strategy)

T-fair-Prompting selects demonstrations based on **individual fairness**.

### Step 1: One-shot evaluation

For each training example $x_i$, compute

$$
\text{fair}(\Gamma(x_i, y_i))
$$

using a single demonstration and the content-free input.

---

### Step 2: Ranking

All demonstrations are ranked in descending order of fairness

$$
\text{fair}(\Gamma(x_{(1)}, y_{(1)}))
\ge
\text{fair}(\Gamma(x_{(2)}, y_{(2)}))
\ge
\cdots
$$

---

### Step 3: Selection

The top $k$ demonstrations are selected to form the final prompt

$$
\rho_{\text{T-fair}}
=
\Gamma(x_{(1)}, y_{(1)})
\oplus
\cdots
\oplus
\Gamma(x_{(k)}, y_{(k)})
$$

### Properties

- Time complexity: $\mathcal{O}(N)$
- Considers demonstrations independently  
- Requires manual choice of $k$

---

## 6. G-fair-Prompting (Greedy Strategy)

G-fair-Prompting constructs the prompt **iteratively**, optimizing **global fairness**.

---

### Initialization

Start with an empty prompt

$$
\rho_0 = \varnothing
$$

---

### Greedy update

At step $t$, select the demonstration that maximizes fairness improvement

$$
(x^*, y^*)
=
\arg\max_{(x_i, y_i) \in \mathcal{S}'}
\text{fair}(\Gamma(x_i, y_i) \oplus \rho_t)
$$

where $\mathcal{S}'$ is the set of remaining unused demonstrations.

The prompt is updated as

$$
\rho_{t+1} = \Gamma(x^*, y^*) \oplus \rho_t
$$

---

### Stopping criterion

The algorithm terminates when no candidate improves fairness

$$
\text{fair}(\rho_{t+1}) \le \text{fair}(\rho_t)
$$

---

### Final prompt

The output is the prompt $\rho_T$ obtained at convergence.

### Properties

- Time complexity: $\mathcal{O}(N^2)$  
- Automatically determines the number of demonstrations  
- Accounts for interactions between demonstrations  

---

## 7. Relationship Between T-fair and G-fair

| Aspect | T-fair | G-fair |
|------|-------|--------|
| Selection style | Independent | Iterative |
| Global interactions | No | Yes |
| Chooses number of examples | No | Yes |
| Approximation quality | Coarse | Near-optimal |

---

## 8. Key Insight

Prompt quality is strongly correlated with **predictive entropy** on content-free input.

Maximizing fairness minimizes inherent prompt bias and leads to more stable and accurate in-context learning.

---

## 9. Summary

- Predictive bias can be measured without labeled validation data
- Entropy provides an efficient fairness surrogate
- T-fair offers a fast baseline
- G-fair provides a higher-quality greedy approximation

These strategies enable principled and efficient few-shot prompt construction for large language models.
