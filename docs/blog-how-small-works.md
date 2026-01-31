# Delta: Lossless Token Sequence Compression for Large Language Models

*A Technical Exposition on Dictionary-Based Compression, Algorithmic Guarantees, and the Economics of Inference Cost Reduction*

---

## Abstract

We present **Delta**, an open-source implementation of Lossless Token Sequence Compression (LTSC) designed to reduce the computational and economic cost of large language model inference. Delta identifies repeated multi-token subsequences within input sequences and replaces them with compact meta-token references backed by a prefix dictionary. The system achieves 30-60% compression on structured inputs while providing mathematical guarantees of perfect reconstruction. This technical exposition details the algorithmic foundations, complexity analysis, and empirical performance characteristics that underpin Delta's design.

---

## 1. Introduction: The Redundancy Problem in Context-Augmented Generation

Modern LLM deployments increasingly rely on context augmentation—the practice of prepending relevant information to input prompts to ground model responses. This encompasses retrieval-augmented generation (RAG), tool schema injection, code repository context, and multi-turn conversation history. While effective for improving output quality, these techniques produce token sequences with substantial redundancy.

Consider the structure of a typical agentic workflow:

$$
\text{Request}_i = \mathcal{S} \oplus \mathcal{T} \oplus \mathcal{C}_i \oplus \mathcal{Q}_i
$$

Where $\mathcal{S}$ denotes the system prompt, $\mathcal{T}$ the tool schemas, $\mathcal{C}_i$ the context for request $i$, and $\mathcal{Q}_i$ the user query. Across $N$ requests in a session, $\mathcal{S}$ and $\mathcal{T}$ are transmitted $N$ times, and $\mathcal{C}_i$ often shares significant overlap with $\mathcal{C}_j$ for $i \neq j$.

The economic implications are substantial. Given a model with input pricing $p$ (USD per million tokens) and average context length $L$ tokens across $R$ requests per day, daily input costs are:

$$
\text{Cost}_{\text{daily}} = \frac{p \cdot L \cdot R}{10^6}
$$

For Claude Opus 4.5 at $p = 5.00$, with $L = 50{,}000$ and $R = 10{,}000$:

$$
\text{Cost}_{\text{daily}} = \frac{5.00 \times 50{,}000 \times 10{,}000}{10^6} = \$2{,}500
$$

This yields monthly costs exceeding $75,000—a significant fraction of which is spent retransmitting identical subsequences.

---

## 2. Theoretical Foundation: The Compressibility Constraint

Delta's core operation is the replacement of repeated token patterns with dictionary references. The fundamental constraint governing this replacement ensures that compression never increases sequence length.

### 2.1 Formal Definition

Let $\pi = (t_1, t_2, \ldots, t_L)$ be a token pattern of length $L$ that occurs at $C$ non-overlapping positions within input sequence $\mathbf{x} = (x_1, x_2, \ldots, x_n)$.

**Definition 2.1 (Original Token Cost).** The original representation consumes:

$$
\text{Cost}_{\text{original}}(\pi, C) = L \cdot C
$$

tokens in the input sequence.

**Definition 2.2 (Compressed Token Cost).** Under dictionary compression with meta-token $\mu_\pi$, the representation requires:

$$
\text{Cost}_{\text{compressed}}(\pi, C) = \underbrace{1}_{\text{meta-token}} + \underbrace{L}_{\text{definition}} + \underbrace{C}_{\text{references}} + \underbrace{\delta}_{\text{overhead}}
$$

where $\delta$ represents format overhead (e.g., length tokens, delimiters).

**Theorem 2.1 (Compressibility Constraint).** Pattern $\pi$ is compressible if and only if:

$$
L \cdot C > 1 + L + C + \delta
$$

*Proof.* Compression is beneficial when $\text{Cost}_{\text{original}} > \text{Cost}_{\text{compressed}}$:
$$
L \cdot C > 1 + L + C + \delta
$$
Rearranging for minimum occurrence count:
$$
C(L - 1) > 1 + L + \delta
$$
$$
C > \frac{1 + L + \delta}{L - 1}
$$
$\square$

**Corollary 2.1 (Minimum Occurrence Count).** The minimum number of occurrences required for compressibility is:

$$
C_{\min}(L, \delta) = \left\lceil \frac{2 + L + \delta}{L - 1} \right\rceil
$$

### 2.2 Compressibility Table

For the default configuration with length tokens enabled ($\delta = 1$):

| Pattern Length $L$ | $C_{\min}$ | Savings per Additional Occurrence |
|--------------------|------------|-----------------------------------|
| 2 | 5 | $L - 1 = 1$ token |
| 3 | 3 | $L - 1 = 2$ tokens |
| 4 | 3 | $L - 1 = 3$ tokens |
| 5 | 2 | $L - 1 = 4$ tokens |
| 8 | 2 | $L - 1 = 7$ tokens |
| 16 | 2 | $L - 1 = 15$ tokens |

**Observation.** Longer patterns amortize dictionary overhead more efficiently, requiring fewer occurrences to achieve compressibility. However, longer patterns are statistically less likely to repeat, creating a fundamental tension that the selection algorithm must navigate.

### 2.3 Net Savings Function

Given a set of selected patterns $\Pi = \{\pi_1, \ldots, \pi_k\}$ with occurrence counts $\{C_1, \ldots, C_k\}$, the total token savings is:

$$
\text{Savings}(\Pi) = \sum_{i=1}^{k} \left[ L_i \cdot C_i - (1 + L_i + C_i + \delta) \right]
$$

$$
= \sum_{i=1}^{k} \left[ (L_i - 1)(C_i - 1) - (2 + \delta) \right]
$$

This formulation reveals that savings grow quadratically with pattern utility $(L_i - 1)(C_i - 1)$, offset by a fixed per-pattern cost of $(2 + \delta)$.

---

## 3. Algorithmic Architecture

Delta's compression pipeline consists of five sequential stages, each with well-defined computational complexity.

### 3.1 Stage 1: Pattern Discovery

The discovery stage identifies all repeated subsequences of length $L \in [L_{\min}, L_{\max}]$ within the input sequence.

#### 3.1.1 Suffix Array Construction

We employ the **doubling algorithm** for suffix array construction, achieving $O(n \log n)$ time complexity.

**Definition 3.1 (Suffix Array).** For sequence $\mathbf{x} = (x_1, \ldots, x_n)$, the suffix array $\text{SA}$ is a permutation of $\{0, 1, \ldots, n-1\}$ such that:

$$
\mathbf{x}[\text{SA}[i]:] <_{\text{lex}} \mathbf{x}[\text{SA}[i+1]:]
$$

for all $i \in \{0, \ldots, n-2\}$, where $<_{\text{lex}}$ denotes lexicographic ordering.

**Definition 3.2 (LCP Array).** The longest common prefix array $\text{LCP}$ satisfies:

$$
\text{LCP}[i] = \text{lcp}(\mathbf{x}[\text{SA}[i]:], \mathbf{x}[\text{SA}[i+1]:])
$$

where $\text{lcp}(\mathbf{a}, \mathbf{b})$ returns the length of the longest common prefix of sequences $\mathbf{a}$ and $\mathbf{b}$.

We compute the LCP array using **Kasai's algorithm** in $O(n)$ time after suffix array construction.

#### 3.1.2 LCP Interval Enumeration

**Definition 3.3 (LCP Interval).** An LCP interval $[i, j]$ with LCP value $\ell$ satisfies:
1. $\text{LCP}[k] \geq \ell$ for all $k \in \{i, \ldots, j-1\}$
2. $\text{LCP}[i-1] < \ell$ (or $i = 0$)
3. $\text{LCP}[j] < \ell$ (or $j = n-1$)

Each LCP interval corresponds to a set of positions sharing a common prefix of length at least $\ell$. We enumerate these intervals using a stack-based algorithm in $O(n)$ time:

```
Algorithm: LCP_INTERVALS(SA, LCP, min_len)
  intervals ← []
  stack ← []
  for i ← 0 to |LCP| - 1:
    start ← i
    while stack ≠ ∅ and stack.top().lcp > LCP[i]:
      (prev_start, prev_lcp) ← stack.pop()
      if prev_lcp ≥ min_len:
        intervals.append((prev_start, i, prev_lcp))
      start ← prev_start
    if stack = ∅ or stack.top().lcp < LCP[i]:
      stack.push((start, LCP[i]))
  while stack ≠ ∅:
    (start, lcp) ← stack.pop()
    if lcp ≥ min_len:
      intervals.append((start, |LCP|, lcp))
  return intervals
```

**Complexity.** Suffix array construction: $O(n \log n)$. LCP computation: $O(n)$. Interval enumeration: $O(n)$. Total discovery: $O(n \log n)$.

#### 3.1.3 Alternative Discovery Strategies

**BPE-Style Iterative Discovery.** For inputs where hierarchical patterns dominate, we implement byte-pair encoding:

```
Algorithm: BPE_DISCOVERY(tokens, max_iterations)
  for iter ← 1 to max_iterations:
    pairs ← count_adjacent_pairs(tokens)
    best_pair ← argmax_{p ∈ pairs} count(p)
    if count(best_pair) < C_min(2, δ):
      break
    tokens ← merge(tokens, best_pair, new_symbol)
  return discovered_patterns
```

Complexity: $O(n \cdot \text{max\_iterations})$.

**Chunked Discovery.** For sequences exceeding memory constraints, we partition into overlapping chunks:

$$
\mathbf{x} = \mathbf{x}_1 \oplus \mathbf{x}_2 \oplus \cdots \oplus \mathbf{x}_m
$$

where $|\mathbf{x}_i| = \text{chunk\_size}$ and consecutive chunks overlap by $\text{overlap\_size}$ tokens. Patterns are discovered independently per chunk, then merged with position reconciliation.

### 3.2 Stage 2: Candidate Filtering

Raw discovery produces a candidate set $\mathcal{P}$ that may contain redundant or non-viable patterns.

#### 3.2.1 Compressibility Pre-filtering

We eliminate patterns that cannot achieve compressibility regardless of selection:

$$
\mathcal{P}' = \left\{ \pi \in \mathcal{P} : |\text{positions}(\pi)| \geq C_{\min}(|\pi|, \delta) \right\}
$$

#### 3.2.2 Subsumption Analysis

**Definition 3.4 (Subsumption).** Pattern $\pi_a$ subsumes pattern $\pi_b$ if $\pi_b$ is a contiguous subsequence of $\pi_a$ and every occurrence of $\pi_b$ overlaps with some occurrence of $\pi_a$.

For subsumed patterns, we compute independent occurrences:

$$
\text{Independent}(\pi_b, \pi_a) = \left\{ p \in \text{positions}(\pi_b) : \forall q \in \text{positions}(\pi_a), [p, p + |\pi_b|) \cap [q, q + |\pi_a|) = \emptyset \right\}
$$

Pattern $\pi_b$ is retained only if $|\text{Independent}(\pi_b, \pi_a)| \geq C_{\min}(|\pi_b|, \delta)$.

### 3.3 Stage 3: Pattern Selection

Given filtered candidates $\mathcal{P}'$, we must select a subset of non-overlapping occurrences that maximizes total savings. This is a variant of the **weighted interval scheduling** problem.

#### 3.3.1 Problem Formulation

Let $\mathcal{O} = \{o_1, \ldots, o_m\}$ be the set of all occurrences, where each $o_i$ is characterized by:
- Start position $s_i$
- End position $e_i = s_i + L_i$
- Pattern $\pi_i$
- Weight $w_i$ (contribution to total savings)

**Objective:** Select $\mathcal{O}^* \subseteq \mathcal{O}$ maximizing:

$$
\sum_{o_i \in \mathcal{O}^*} w_i
$$

**Subject to:** Non-overlap constraint:

$$
\forall o_i, o_j \in \mathcal{O}^*, i \neq j: [s_i, e_i) \cap [s_j, e_j) = \emptyset
$$

**Additional constraint:** Compressibility—each selected pattern must have enough selected occurrences:

$$
\forall \pi: |\{o_i \in \mathcal{O}^* : \pi_i = \pi\}| \geq C_{\min}(|\pi|, \delta) \text{ or } |\{o_i \in \mathcal{O}^* : \pi_i = \pi\}| = 0
$$

#### 3.3.2 Weight Computation

The weight of an occurrence depends on the expected count of its pattern after selection. We use **amortized dictionary cost**:

$$
w_i = (L_i - 1) - \frac{1 + L_i + \delta}{\hat{C}_i}
$$

where $\hat{C}_i$ is the estimated non-overlapping occurrence count for pattern $\pi_i$, computed via greedy sweep:

```
Algorithm: ESTIMATE_NON_OVERLAPPING(positions, length)
  count ← 0
  next_free ← -∞
  for p in sorted(positions):
    if p ≥ next_free:
      count ← count + 1
      next_free ← p + length
  return count
```

#### 3.3.3 Selection Algorithms

**Greedy Selection ($O(m \log m)$).**

We define a **savings density** heuristic:

$$
\text{density}(o_i) = \frac{L_i - 1}{L_i} + \alpha \cdot \text{priority}_i
$$

where $\alpha = 0.1$ balances length-based savings with external priority signals.

```
Algorithm: GREEDY_SELECT(occurrences)
  sort occurrences by density descending
  selected ← []
  occupied ← ∅
  for o in occurrences:
    if positions(o) ∩ occupied = ∅:
      selected.append(o)
      occupied ← occupied ∪ positions(o)
  return REFINE_FOR_COMPRESSIBILITY(selected)
```

The refinement loop iteratively removes patterns that failed to achieve compressibility, freeing positions for other candidates:

```
Algorithm: REFINE_FOR_COMPRESSIBILITY(selected)
  for iteration ← 1 to max_iterations:
    pattern_counts ← count patterns in selected
    non_compressible ← {π : count(π) < C_min(|π|, δ)}
    if non_compressible = ∅:
      break
    selected ← [o ∈ selected : π(o) ∉ non_compressible]
    // Re-run selection with freed positions
    selected ← GREEDY_SELECT_INCREMENTAL(selected, freed_positions)
  return selected
```

**Optimal Selection via Dynamic Programming ($O(m \log m)$).**

Sort occurrences by end position. For each $o_i$, compute $p_i$ = largest index $j < i$ such that $e_j \leq s_i$ (via binary search).

Recurrence:

$$
\text{OPT}[i] = \max\left( \text{OPT}[i-1], w_i + \text{OPT}[p_i] \right)
$$

Backtrack to recover selected occurrences.

**Beam Search ($O(m \cdot W)$).**

Maintain $W$ candidate selections (beam width). For each occurrence, extend each candidate by either including or excluding, keeping top $W$ by cumulative savings.

Beam search uses **marginal savings** for scoring:

$$
\text{marginal}(o_i, C_{\text{current}}) = \text{Savings}(L_i, C_{\text{current}} + 1) - \text{Savings}(L_i, C_{\text{current}})
$$

**Integer Linear Programming (Exponential, Optimal).**

Formulate as ILP with binary decision variables $x_i \in \{0, 1\}$:

$$
\max \sum_i w_i \cdot x_i
$$

Subject to:

$$
\sum_{i : o_i \text{ covers position } p} x_i \leq 1 \quad \forall p \in \{0, \ldots, n-1\}
$$

Solved via scipy's MILP solver with configurable timeout.

### 3.4 Stage 4: Hierarchical Compression

After initial compression, the output sequence may contain new repeated patterns formed by meta-token combinations. We apply compression recursively:

$$
\mathbf{x}^{(0)} = \mathbf{x}, \quad \mathbf{x}^{(k+1)} = \text{Compress}(\mathbf{x}^{(k)})
$$

**Early Stopping Criteria.** We terminate when:

1. **Improvement ratio** drops below threshold $\tau_{\text{imp}} = 0.02$:

$$
\frac{|\mathbf{x}^{(k)}| - |\mathbf{x}^{(k+1)}|}{|\mathbf{x}^{(k)}|} < \tau_{\text{imp}}
$$

2. **Efficiency ratio** drops below threshold $\tau_{\text{eff}} = 1.5$:

$$
\frac{\text{body\_savings}}{\text{dictionary\_growth}} < \tau_{\text{eff}}
$$

3. **Maximum depth** $d_{\max} = 3$ is reached.

### 3.5 Stage 5: Serialization and Verification

The output format is:

```
[<StaticDict:domain>]  // Optional static dictionary marker
<Dict>
<MT_0> <Len:L_0> t_{0,1} t_{0,2} ... t_{0,L_0}
<MT_1> <Len:L_1> t_{1,1} t_{1,2} ... t_{1,L_1}
...
</Dict>
BODY_TOKENS
```

**Round-trip Verification.** When enabled, we assert:

$$
\text{Decompress}(\text{Compress}(\mathbf{x})) = \mathbf{x}
$$

This provides a runtime guarantee of lossless reconstruction.

---

## 4. Advanced Capabilities

### 4.1 Static Dictionaries

For domain-specific content, pre-computed dictionaries bypass discovery entirely. Let $\mathcal{D}_{\text{static}} = \{(\mu_i, \pi_i)\}$ be a static dictionary. Before dynamic compression:

$$
\mathbf{x}' = \text{ApplyStaticDict}(\mathbf{x}, \mathcal{D}_{\text{static}})
$$

Available dictionaries: `python-v1`, `typescript-v1`, `json-v1`, `sql-v1`, `markdown-v1`.

Domain detection uses heuristic scoring:

$$
\text{score}(d) = \sum_{\pi \in \mathcal{D}_d} \mathbb{1}[\pi \subseteq \mathbf{x}] \cdot |\pi|
$$

The domain with highest score above confidence threshold $\gamma = 0.85$ is selected.

### 4.2 Region-Aware Compression

Different regions of a context window have different semantic importance. We define region types $\mathcal{R} = \{\text{SYSTEM}, \text{USER}, \text{ASSISTANT}, \text{CONTEXT}, \text{CODE}, \text{DATA}\}$ with associated compression limits:

| Region | Max Compression | Priority Adjustment |
|--------|-----------------|---------------------|
| SYSTEM | 5% | -2 |
| USER | 10% | -1 |
| ASSISTANT | 15% | 0 |
| CONTEXT | 50% | +3 |
| DATA | 60% | +4 |

Candidates spanning protected regions receive priority penalties proportional to overlap.

### 4.3 Pattern Importance Scoring

We implement a composite importance scorer:

$$
I(\pi) = \alpha_{\text{pos}} \cdot I_{\text{pos}}(\pi) + \alpha_{\text{freq}} \cdot I_{\text{freq}}(\pi) + \alpha_{\text{len}} \cdot I_{\text{len}}(\pi)
$$

Where:

- $I_{\text{pos}}(\pi) = 1 - \frac{\text{mean\_position}(\pi)}{n}$ (early patterns more important)
- $I_{\text{freq}}(\pi) = 1 - \frac{\log(1 + \text{count}(\pi))}{\log(1 + \max\_\text{count})}$ (rare patterns more important)
- $I_{\text{len}}(\pi) = \frac{|\pi| - L_{\min}}{L_{\max} - L_{\min}}$ (longer patterns potentially structural)

Patterns with $I(\pi) > \theta_{\text{importance}}$ are filtered from compression.

### 4.4 Cross-Document Pattern Cache

For workloads with document similarity, we maintain a persistent cache:

$$
\mathcal{C} = \{(\pi, f_\pi, t_\pi)\}
$$

where $f_\pi$ is frequency and $t_\pi$ is last-seen timestamp.

**Warm-start injection:** Top-$k$ cached patterns are injected as preferred candidates.

**Frequency decay:** $f_\pi \leftarrow f_\pi \cdot 2^{-\Delta t / \tau_{1/2}}$ where $\tau_{1/2} = 100$ compressions.

### 4.5 Streaming Compression

For unbounded inputs, we process in chunks with overlap:

$$
\mathbf{x} = \bigoplus_{i=1}^{m} \mathbf{x}_i, \quad |\mathbf{x}_i| = \text{chunk\_size}, \quad |\mathbf{x}_i \cap \mathbf{x}_{i+1}| = \text{overlap}
$$

Each chunk is compressed independently with its own dictionary. The pattern cache propagates learned patterns across chunks.

**Memory bound:** $O(\text{chunk\_size} + \text{overlap})$ regardless of total input length.

---

## 5. Formal Guarantees

**Theorem 5.1 (Lossless Reconstruction).** For any input sequence $\mathbf{x}$ and configuration $\mathcal{C}$:

$$
\text{Decompress}(\text{Compress}(\mathbf{x}, \mathcal{C}), \mathcal{C}) = \mathbf{x}
$$

*Proof.* The dictionary format is unambiguous: each meta-token maps to exactly one subsequence. Decompression recursively expands meta-tokens until none remain. The compressibility constraint ensures the compressed representation is never longer than the original, so compression is always valid. $\square$

**Theorem 5.2 (Non-Expansion Guarantee).** For any input sequence $\mathbf{x}$:

$$
|\text{Compress}(\mathbf{x})| \leq |\mathbf{x}|
$$

*Proof.* If no compressible patterns exist, we return $\mathbf{x}$ unchanged. If patterns exist, each is compressed only if it satisfies the compressibility constraint, guaranteeing net savings. A final length check returns the shorter of compressed and original. $\square$

**Theorem 5.3 (Determinism).** For fixed input $\mathbf{x}$ and configuration $\mathcal{C}$:

$$
\text{Compress}(\mathbf{x}, \mathcal{C}) = \text{Compress}(\mathbf{x}, \mathcal{C})
$$

*Proof.* All algorithms use deterministic sorting and selection. The optional RNG seed ensures reproducibility when randomization is used. $\square$

---

## 6. Complexity Analysis

| Stage | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Suffix Array Construction | $O(n \log n)$ | $O(n)$ |
| LCP Computation | $O(n)$ | $O(n)$ |
| LCP Interval Enumeration | $O(n)$ | $O(n)$ |
| Candidate Filtering | $O(p)$ | $O(p)$ |
| Greedy Selection | $O(m \log m)$ | $O(m)$ |
| Optimal Selection (DP) | $O(m \log m)$ | $O(m)$ |
| Beam Search | $O(m \cdot W)$ | $O(m \cdot W)$ |
| Hierarchical Passes | $O(d \cdot n \log n)$ | $O(n)$ |
| **Total (Greedy)** | $O(n \log n)$ | $O(n)$ |

Where $n$ = input length, $p$ = candidate patterns, $m$ = total occurrences, $W$ = beam width, $d$ = hierarchical depth.

---

## 7. Empirical Performance

### 7.1 Compression Ratios

| Input Type | Typical Savings | Characteristics |
|------------|-----------------|-----------------|
| Highly repetitive | 50-70% | Same pattern $>50$ times |
| Structured code | 35-50% | Repeated function signatures, imports |
| RAG with overlap | 30-45% | Retrieved chunks share structure |
| Multi-turn conversation | 25-40% | Prior turns in context |
| Varied natural language | 5-20% | Limited structural repetition |
| Random/unique tokens | 0% | No compressible patterns |

### 7.2 Latency Benchmarks

Measured on Apple M2 Pro, single-threaded:

| Input Size (tokens) | Python | TypeScript/WASM |
|---------------------|--------|-----------------|
| 1,000 | 5ms | 2ms |
| 8,000 | 40ms | 8ms |
| 32,000 | 150ms | 25ms |
| 128,000 | 600ms | 100ms |

### 7.3 Throughput

$$
\text{Throughput} = \frac{n}{\text{latency}} \approx 200{,}000 \text{ tokens/sec (Python)}, \; 1{,}280{,}000 \text{ tokens/sec (WASM)}
$$

---

## 8. Economic Analysis

### 8.1 Cost Model

Let:
- $p$ = price per million input tokens
- $L$ = average context length
- $R$ = requests per day
- $\rho$ = compression ratio (compressed / original)

Daily cost without compression:

$$
\text{Cost}_{\text{base}} = \frac{p \cdot L \cdot R}{10^6}
$$

Daily cost with compression:

$$
\text{Cost}_{\text{compressed}} = \frac{p \cdot \rho \cdot L \cdot R}{10^6}
$$

Daily savings:

$$
\text{Savings}_{\text{daily}} = (1 - \rho) \cdot \text{Cost}_{\text{base}}
$$

### 8.2 Example Scenarios

**Scenario A: Agentic Coding Workflow**

| Parameter | Value |
|-----------|-------|
| Model | Claude Opus 4.5 |
| Price $p$ | $5.00/M tokens |
| Context $L$ | 50,000 tokens |
| Requests/day $R$ | 10,000 |
| Compression ratio $\rho$ | 0.65 (35% savings) |

$$
\text{Savings}_{\text{monthly}} = (1 - 0.65) \times \frac{5.00 \times 50{,}000 \times 10{,}000}{10^6} \times 30 = \$26{,}250
$$

**Scenario B: High-Volume RAG System**

| Parameter | Value |
|-----------|-------|
| Model | GPT-5.2 |
| Price $p$ | $1.75/M tokens |
| Context $L$ | 100,000 tokens |
| Requests/day $R$ | 100,000 |
| Compression ratio $\rho$ | 0.50 (50% savings) |

$$
\text{Savings}_{\text{monthly}} = 0.50 \times \frac{1.75 \times 100{,}000 \times 100{,}000}{10^6} \times 30 = \$262{,}500
$$

### 8.3 Break-Even Analysis

The compression overhead (CPU time) is negligible relative to API latency. For an 8K context with 40ms compression time versus 2-5 second inference latency:

$$
\text{Overhead ratio} = \frac{40\text{ms}}{2000\text{ms}} = 2\%
$$

Break-even occurs at extremely low compression ratios:

$$
\rho_{\text{break-even}} = 1 - \frac{\text{compute\_cost}}{\text{API\_cost}} \approx 0.99
$$

Any measurable compression provides net economic benefit.

---

## 9. Why We Open-Sourced Delta

The decision to release Delta as open-source software reflects our perspective on infrastructure-level tooling and the current state of AI deployment economics.

### 9.1 The Inference Cost Bottleneck

Inference cost remains one of the most underappreciated constraints on AI adoption. Teams routinely make architectural compromises to manage token budgets: truncating context windows, limiting agent autonomy, reducing RAG retrieval depth, or avoiding multi-turn interactions entirely. These compromises directly impact capability.

The constraint is particularly acute for agentic workflows, where context windows balloon quickly. A coding agent that maintains file context, tool schemas, and conversation history can easily consume 100K+ tokens per interaction. At current pricing, this becomes economically prohibitive at scale.

### 9.2 Why Not Keep It Proprietary?

We considered offering Delta as a paid service. The decision against this came down to several factors:

**Deployment topology.** Compression must occur before tokens reach the inference API. This means the compressor sits in the critical path of every request. Adding a network hop to a third-party service introduces latency, creates a single point of failure, and raises data handling concerns for sensitive workloads. The right deployment model is client-side, which favors open distribution.

**Adoption dynamics.** Compression utilities are most valuable when widely adopted. If compression becomes a standard preprocessing step, model providers can optimize for compressed inputs, tokenizers can account for dictionary formats, and the entire ecosystem benefits. Proprietary tooling fragments this potential.

**Competitive moat.** The algorithms underlying Delta are not novel—suffix arrays, weighted interval scheduling, and dictionary compression are well-established techniques. Our value-add is a production-quality implementation with comprehensive configuration, robust error handling, and multi-platform support. This is difficult to sustain as a proprietary advantage.

**Strategic alignment.** Triage's core business is AI observability and security tooling. Reducing inference costs for the broader ecosystem expands the addressable market for our primary products. Delta is infrastructure that benefits everyone building AI systems, including our customers.

### 9.3 The Ecosystem Benefit

We believe that inference cost reduction should not be a competitive differentiator—it should be table stakes. Every team deploying LLMs faces the same redundancy problem. The solution is architectural, not proprietary.

By releasing Delta openly, we aim to:

1. **Lower barriers to AI adoption.** Teams with limited budgets can deploy more capable systems.
2. **Enable architectural experimentation.** Reduced inference costs make longer context windows, deeper RAG retrieval, and more autonomous agents economically viable.
3. **Establish interoperability standards.** A common compression format enables cross-tool compatibility and provider-level optimizations.
4. **Accelerate research.** Compression-aware training and evaluation require accessible tooling.

### 9.4 Sustainability Model

Open-source does not mean unsupported. We maintain Delta as part of our broader infrastructure investment and welcome contributions from the community. Enterprise support, custom integrations, and compression-aware observability are available through Triage's commercial offerings.

---

## 10. Conclusion

Delta provides a mathematically grounded, production-ready approach to reducing LLM inference costs through lossless token compression. The system achieves 30-60% compression on structured inputs while guaranteeing perfect reconstruction, with sub-50ms latency for typical context windows.

The core insight is simple: context-augmented generation produces redundant token sequences, and this redundancy can be systematically eliminated at the compression layer without semantic loss. The implementation details—suffix array discovery, weighted interval scheduling, hierarchical compression, and cross-document caching—are engineering refinements on this fundamental observation.

We release Delta as open-source software because we believe infrastructure-level cost reduction benefits the entire AI ecosystem. Inference cost should not be a barrier to building capable AI systems.

---

## References

1. Harvill, J., et al. (2024). "Lossless Token Sequence Compression via Meta-Tokens."
2. Kasai, T., et al. (2001). "Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its Applications."
3. Manber, U., & Myers, G. (1993). "Suffix Arrays: A New Method for On-Line String Searches."
4. Kleinberg, J., & Tardos, E. (2006). "Algorithm Design." Chapter 6: Weighted Interval Scheduling.

---

## Appendix A: Configuration Reference

```python
CompressionConfig(
    # Pattern discovery bounds
    min_subsequence_length: int = 2,      # L_min
    max_subsequence_length: int = 8,      # L_max

    # Discovery algorithm: "suffix-array" | "sliding-window" | "bpe"
    discovery_mode: str = "suffix-array",

    # Selection algorithm: "greedy" | "optimal" | "beam" | "ilp" | "semantic"
    selection_mode: str = "greedy",
    beam_width: int = 8,                  # W for beam search

    # Hierarchical compression
    hierarchical_enabled: bool = True,
    hierarchical_max_depth: int = 3,      # d_max
    hierarchical_min_improvement: float = 0.02,  # τ_imp

    # Dictionary format
    dict_length_enabled: bool = True,     # Adds δ = 1 overhead

    # Verification
    verify: bool = False,                 # Enable round-trip verification
)
```

---

## Appendix B: Serialization Format Grammar

```
compressed_sequence := [static_marker] dictionary body

static_marker := "<StaticDict:" domain ">"

dictionary := "<Dict>" entry* "</Dict>"

entry := meta_token length_token token+

meta_token := "<MT_" integer ">"

length_token := "<Len:" integer ">"

body := (token | meta_token | patch_sequence)*

patch_sequence := meta_token "<Patch>" patch_entry+ "</Patch>"

patch_entry := "<Idx:" integer ">" token
```

---

*Delta is available under the MIT License at [github.com/delta-ltsc/delta](https://github.com/delta-ltsc/delta).*

*PyPI: [theta-ltsc](https://pypi.org/project/theta-ltsc/) | npm: [@theta-ltsc/sdk](https://www.npmjs.com/package/@theta-ltsc/sdk)*
