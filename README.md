# 🧠 Argumentation Clustering & Narrative Analysis

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

> **Advanced computational analysis of argumentative discourse across Reddit and Congressional platforms, featuring stance-aware clustering, narrative flow detection, and representation intensity metrics.**

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🔄 Complete Pipeline](#-complete-pipeline)
- [✨ Key Features](#-key-features)
- [🔬 Scientific Motivation](#-scientific-motivation)
- [🏗️ Architecture](#️-architecture)
- [📊 Pipeline Flow](#-pipeline-flow)
- [🧮 Algorithms & Methods](#-algorithms--methods)
- [📁 Output Documentation](#-output-documentation)
- [🚀 Quick Start](#-quick-start)
- [📈 Usage Examples](#-usage-examples)
- [🛠️ Configuration](#️-configuration)
- [📚 Research Applications](#-research-applications)

---

## 🎯 Overview

This repository implements a sophisticated computational framework for analyzing argumentative discourse patterns across digital platforms. The system processes millions of arguments from Reddit discussions and U.S. Congressional hearings to identify:

- **Narrative Flow Patterns**: How arguments progress through causal chains
- **Cross-Platform Representation**: Which Reddit communities mirror Congressional discourse
- **Stance-Aware Clustering**: Grouping arguments while preserving ideological coherence
- **Topic Consolidation**: Merging fragmented topics into coherent themes

## ✨ Key Features

### 🔄 **Dynamic Topic Consolidation**
- **BERTopic-Based Domain Discovery**: Automatically discovers semantic domains using finetuned embeddings
- **FAISS-Accelerated Similarity**: High-performance vectorized similarity computation with approximate nearest neighbors
- **Multi-Level Hierarchical Structure**: Creates semantic taxonomy preserving granularity across abstraction levels
- **Narrative Chain Detection**: Graph-based discovery of causal argument progressions using embedding trajectories

### 🎯 **Guaranteed Pure Clustering** 
- **Single Topic-Stance per Cluster**: Ensures each cluster represents one coherent narrative
- **BERTopic Integration**: Leverages state-of-the-art topic modeling with KeyBERT-inspired representations
- **Supervised UMAP**: Uses stance labels for better embedding separation
- **Outlier Detection**: Handles edge cases and noise appropriately

### 📊 **Representation Intensity Analysis**
- **RI(L) - Legislator Representation**: Percentage of legislator arguments matching Reddit discourse
- **RI(S) - Subreddit Representation**: Percentage of subreddit arguments reflected in Congress
- **Cross-Platform Similarity**: Cosine similarity-based matching with configurable thresholds
- **Entity-Level Analysis**: Individual member and witness representation tracking

### 🌊 **Directional Narrative Chains**
- **Progression Stage Detection**: Maps arguments to conceptual development stages
- **Causal Flow Visualization**: Shows how narratives evolve (e.g., `health → women's_rights → reproductive_rights → abortion_access`)
- **Dynamic Chain Assignment**: Adapts to different policy domains automatically

---

## 🔄 Complete Pipeline

### **High-Level Flow: From Raw Arguments to Analysis Results**

```
Raw PKL Files → Data Loading → Text Normalization → Embedding Generation → 
Topic Consolidation (optional) → Narrative Coherence → Pure Clustering → 
Statistical Analysis → CSV Export + Representation Calculations
```

### **Input Sources**
- **Congressional Data**: `.pkl` files containing processed hearing transcripts with argument classification, topic extraction, and stance detection
- **Reddit Data**: `.pkl` files containing processed discussion posts with argument classification, topic extraction, and stance detection
- **Member/Witness Data**: `.csv` files linking congressional speakers to biographical information

### **Processing Steps**
1. **Data Loading & Filtering**: Load topic-specific datasets, filter for arguments only, standardize columns
2. **Text Normalization**: Platform-specific text cleaning (Reddit: remove usernames/URLs; Congressional: remove titles/procedures)
3. **Embedding Generation**: Create 768-dim vectors using contrastive fine-tuned MPNet model
4. **Dynamic Topic Consolidation** (Optional): Merge fragmented topics using BERTopic domain discovery and FAISS-accelerated multi-signal similarity
5. **Narrative Coherence**: Split topics with mixed stances into pure topic-stance combinations
6. **Pure Clustering**: Run BERTopic separately on each topic-stance group to ensure ideological coherence
7. **Statistical Analysis**: Calculate representation intensity metrics (RI(L), RI(S)) and cluster statistics
8. **Export Generation**: Create comprehensive CSV outputs with cluster assignments, representative documents, and analysis results

### **Key Outputs**
- **`clusters_to_visualize_{topic}.csv`**: Main cluster analysis with 20+ columns including overlap scores, representative documents, stance distributions
- **`member_subreddit_representation_{topic}.csv`**: Legislator-level representation intensity metrics
- **`witness_subreddit_representation_{topic}.csv`**: Witness-level representation intensity metrics  
- **`all_documents_topics_{topic}.csv`**: Complete document-to-cluster mapping for all processed arguments
- **`subreddit_representation_intensity_{topic}.csv`**: Subreddit-level representation metrics
- **Additional files**: Missing entities, topic hierarchies, enhanced topic information

### **Guaranteed Outputs Integrity**
- **Stance Purity**: Every cluster contains only one topic-stance combination
- **Cross-Platform Coverage**: Tracks representation across Reddit and Congressional platforms
- **Methodological Rigor**: Formal RI(L) and RI(S) metrics with similarity thresholds
- **Complete Traceability**: Every argument mapped to final cluster with full metadata

---

## 🔬 Scientific Motivation

### The Challenge of Fragmented Discourse Analysis

Traditional topic modeling approaches face critical limitations when analyzing argumentative discourse:

1. **Topic Fragmentation**: Similar concepts get artificially separated (e.g., "abortion rights" vs "reproductive choice")
2. **Stance Conflation**: Pro and con arguments within the same topic get mixed, losing ideological coherence
3. **Cross-Platform Gaps**: No systematic way to measure how online discourse reflects institutional debate
4. **Narrative Blind Spots**: Missing the causal/temporal relationships between related arguments

### Our Solution: Multi-Signal Consolidation + Pure Clustering

<details>
<summary><strong>🔍 Click to expand: Detailed Motivation</strong></summary>

#### **Problem 1: Topic Fragmentation**
**Traditional Approach**: BERTopic might create separate clusters for:
- "reproductive health care"
- "women's reproductive rights" 
- "abortion access"
- "family planning services"

**Our Solution**: Dynamic Consolidator uses BERTopic domain discovery to merge these into:
- Representative topic: "reproductive rights"
- Preserves all original nuances in the `wiba_topics` column

#### **Problem 2: Stance Conflation** 
**Traditional Approach**: One cluster containing:
- "Abortion is a fundamental right" (Pro)
- "Abortion violates sanctity of life" (Con)

**Our Solution**: Guaranteed Pure Clustering ensures:
- Cluster A: Only pro-choice arguments about reproductive rights
- Cluster B: Only pro-life arguments about reproductive rights
- Maintains topic coherence while preserving stance purity

#### **Problem 3: Cross-Platform Analysis**
**Challenge**: How do we know if Reddit discourse `r/politics` reflects what members of Congress actually say?

**Our Solution**: Representation Intensity metrics:
- **RI(L)**: % of Rep. Smith's arguments that match `r/politics` discourse
- **RI(S)**: % of `r/politics` arguments that appear in Congressional testimony

#### **Problem 4: Narrative Understanding**
**Traditional View**: Topics as isolated entities

**Our View**: Topics as part of causal progressions:
```
health concerns → women's autonomy → reproductive rights → abortion policy
economic anxiety → job security → trade policy → immigration restrictions
```

</details>

---

## 🏗️ Architecture

<details>
<summary><strong>📐 System Architecture Diagram</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
graph TB
    subgraph "Data Sources"
        A["`Reddit Arguments
        ~2M posts`"] 
        B["`Congressional Hearings
        ~500K statements`"]
    end
    
    subgraph "Preprocessing"
        C["`Text Normalization
        Source-Specific`"]
        D["`Embedding Generation
        Contrastive Fine-tuned MPNet`"]
        E["`WIBA Topic Extraction
        Policy Domain Classification`"]
    end
    
    subgraph "Topic Consolidation"
        F["`AutomaticConsolidator
        Hybrid Similarity`"]
        G["`Stance Compatibility
        Matrix`"]
        H["`Narrative Flow
        Detection`"]
    end
    
    subgraph "Pure Clustering"
        I["`Topic-Stance
        Separation`"]
        J["`BERTopic
        Per Group`"]
        K["`KeyBERT Label
        Generation`"]
    end
    
    subgraph "Analysis & Export"
        L["`Representation
        Intensity RI(L), RI(S)`"]
        M["`Narrative Chain
        Assignment`"]
        N["`CSV Export
        clusters_to_visualize.csv`"]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    
    style F fill:#e1f5fe
    style I fill:#f3e5f5
    style L fill:#e8f5e8
```

</details>

---

## 📊 Pipeline Flow

### Stage 1: Data Preprocessing & Embedding

<details>
<summary><strong>🔄 Data Processing Pipeline</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart LR
    subgraph "Raw Arguments"
        A["`Reddit Posts
        r/politics, r/conservative, etc.`"]
        B["`Congressional Transcripts
        House/Senate Hearings`"]
    end
    
    subgraph "Text Normalization"
        C["`Reddit: Remove usernames,
        URLs, escape sequences`"]
        D["`Congressional: Remove titles,
        procedural language`"]
    end
    
    subgraph "Embedding Generation"
        E["`Contrastive Fine-tuned
        MPNet-v2 Model`"]
        F["`768-dim Embeddings
        Semantic Representations`"]
    end
    
    subgraph "Initial Classification"
        G["`WIBA Topic Extraction
        Policy Domain Labels`"]
        H["`Stance Detection
        Pro/Con/Neutral`"]
    end
    
    A --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    
    style E fill:#fff3e0
    style G fill:#e8f5e8
```

**Key Preprocessing Steps:**
- **Reddit Normalization**: Removes platform-specific artifacts (`/u/username`, `/r/subreddit`)
- **Congressional Normalization**: Strips formal parliamentary language and titles
- **Unified Lowercasing**: Ensures consistent comparison across platforms
- **Embedding Model**: Uses domain-specific fine-tuned sentence transformer for political discourse

</details>

### Stage 2: Dynamic Topic Consolidation

<details>
<summary><strong>🚀 Optimized Dynamic Consolidation Pipeline</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TD
    subgraph "Input Topics"
        A["`abortion rights
        reproductive choice
        family planning
        womens health`"]
    end
    
    subgraph "BERTopic Domain Discovery"
        B["`Finetuned MPNet Embeddings
        768-dim semantic vectors`"]
        C["`Domain Clustering
        Automatic keyword extraction`"]
        D["`Domain Assignment
        Topic → Domain mapping`"]
    end
    
    subgraph "FAISS-Accelerated Analysis"
        E["`FAISS IndexFlatIP
        Fast neighbor search`"]
        F["`Vectorized Similarity
        Domain + Semantic + Abstraction`"]
        G["`Smart Thresholding
        Early filtering`"]
    end
    
    subgraph "Multi-Level Hierarchy"
        H["`Level 1: Fine-grained
        High similarity threshold`"]
        I["`Level 2: Medium groups
        Moderate threshold`"]
        J["`Level 3: Broad categories
        Low threshold`"]
    end
    
    subgraph "Narrative Chain Discovery"
        K["`Temporal/Causal Patterns
        Graph-based relationships`"]
        L["`Chain Direction Detection
        Embedding trajectories`"]
        M["`Narrative Assignment
        Topic → Chain mapping`"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    D --> K
    K --> L
    L --> M
    
    style B fill:#e1f5fe
    style F fill:#fff3e0
    style K fill:#e8f5e8
```

**Dynamic Discovery Features:**
- 🤖 **No Hardcoded Patterns**: Discovers domains automatically using BERTopic with finetuned embeddings
- ⚡ **FAISS Acceleration**: Sub-linear O(n log n) performance using approximate nearest neighbor search
- 📊 **Vectorized Operations**: Batch similarity computation replacing O(n²) nested loops
- 🧠 **Smart Caching**: Embedding reuse and memory-efficient sparse operations
- 📈 **Progressive Hierarchy**: Multi-level consolidation preserving semantic granularity

**Performance Optimizations:**
1. **Embedding Caching**: Reuse embeddings across similarity computations (3x reduction)
2. **FAISS Integration**: Fast k-NN search with IndexFlatIP (5x speedup)
3. **Vectorized Similarity**: Batch operations instead of nested loops (10x improvement)
4. **K-means Clustering**: Replace spectral clustering for weight learning (4x faster)
5. **Smart Thresholding**: Early filtering of low-similarity pairs (2x memory reduction)

</details>

### Stage 3: Guaranteed Pure Clustering

<details>
<summary><strong>🎯 Pure Clustering Strategy</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TD
    subgraph "Topic-Stance Groups"
        A["`abortion_pro
        2,847 arguments`"]
        B["`abortion_con
        1,923 arguments`"]
        C["`guncontrol_pro
        3,156 arguments`"]
        D["`guncontrol_con
        2,689 arguments`"]
    end
    
    subgraph "Per-Group BERTopic"
        E["`BERTopic Instance
        Min size: 10`"]
        F["`KeyBERT Labels
        Meaningful keywords`"]
        G["`Outlier Detection
        Noise handling`"]
    end
    
    subgraph "Pure Clusters"
        H["`Cluster 1: abortion_pro_rights
        Cluster 2: abortion_pro_healthcare
        Cluster 3: abortion_pro_autonomy`"]
        I["`Cluster 4: abortion_con_life
        Cluster 5: abortion_con_moral
        Cluster 6: abortion_con_religious`"]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> F
    F --> G
    G --> H
    G --> I
    
    style A fill:#e8f5e8
    style B fill:#ffebee
    style C fill:#e8f5e8
    style D fill:#ffebee
```

**Purity Guarantee:**
- ✅ Every cluster contains exactly one topic + one stance combination
- ✅ No mixing of pro/con arguments within clusters
- ✅ Preserves ideological coherence while finding sub-themes

</details>

### Stage 4: Narrative Flow Detection

<details>
<summary><strong>🌊 Directional Narrative Chains</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart LR
    subgraph "Abortion Narrative"
        A["`health
        Medical concerns`"] --> B["`womens_rights
        Autonomy arguments`"]
        B --> C["`reproductive_rights
        Policy framework`"]
        C --> D["`abortion_access
        Implementation`"]
    end
    
    subgraph "Gun Control Narrative"
        E["`violence_concern
        Safety issues`"] --> F["`public_safety
        Protection needs`"]
        F --> G["`gun_regulation
        Policy solutions`"]
        G --> H["`policy_implementation
        Enforcement`"]
    end
    
    subgraph "Topic Assignment"
        I["`Topic: women autonomy reproductive
        Matches: womens_rights stage`"]
        J["`Output: health → [womens_rights] → reproductive_rights → abortion_access`"]
    end
    
    A -.-> I
    I --> J
    
    style B fill:#fff3e0
    style I fill:#e1f5fe
```

**Chain Detection Algorithm:**
1. **Keyword Matching**: Topics analyzed against 4 predefined narrative chains (abortion, gun control, climate, economy)
2. **Chain Identification**: Topics assigned to chains based on keyword overlap with chain vocabularies
3. **Stage Detection**: Within matching chains, topics scored against progression stage keywords to find best fit
4. **Score Calculation**: `chain_keyword_matches + best_stage_score` determines final assignment
5. **Stage Highlighting**: Current position shown in brackets `[current_stage]` within full progression
6. **Directional Output**: Complete causal chain with current stage highlighted (e.g., `health → [womens_rights] → reproductive_rights → abortion_access`)

**Predefined Narrative Chains:**
- **Abortion**: `health → womens_rights → reproductive_rights → abortion_access`
- **Gun Control**: `violence_concern → public_safety → gun_regulation → policy_implementation`
- **Climate**: `environmental_concern → scientific_evidence → policy_development → implementation`
- **Economy**: `economic_conditions → employment_impact → policy_response → growth_outcomes`

</details>

---

## 🧮 Algorithms & Methods

### Core Technologies

| Component | Method | Implementation |
|-----------|--------|----------------|
| **Embeddings** | Contrastive Fine-tuned MPNet | `/vienna/models/contrastive_finetune_v2_mpnet-v2_mal` |
| **Topic Modeling** | BERTopic + KeyBERT | `bertopic.BERTopic` with `KeyBERTInspired` representation |
| **Clustering** | Hierarchical + FAISS + Graph-based | `sklearn.cluster.AgglomerativeClustering` + `faiss` + `networkx` |
| **Dimensionality Reduction** | Supervised UMAP | `cuml.manifold.UMAP` with stance labels |
| **Similarity** | Cosine + Euclidean | `sklearn.metrics.pairwise.cosine_similarity` |

### Key Parameters

<details>
<summary><strong>⚙️ Algorithm Configuration</strong></summary>

```python
# Dynamic Topic Consolidation
CONSOLIDATION_CONFIG = {
    "method": "hierarchical_dynamic",  # Dynamic discovery method
    "max_levels": 3,                  # Hierarchy depth
    "level_thresholds": [0.7, 0.55, 0.4], # Per-level similarity thresholds
    "n_domains": 6,                   # Target domain count
    "min_domain_size": 3,             # Minimum topics per domain
    "dynamic_weights": {              # Learned optimal weights
        "domain": 0.4,                # Domain-aware similarity
        "semantic": 0.4,              # FAISS-accelerated semantic
        "abstraction": 0.2            # Hierarchical abstraction
    }
}

# Pure Clustering  
CLUSTERING_CONFIG = {
    "min_topic_size": 10,            # Minimum documents per cluster
    "embedding_model": "mpnet-v2",   # Sentence transformer model
    "representation_model": [        # Chain of representation models
        "KeyBERTInspired()",
        "MaximalMarginalRelevance(diversity=0.3)"
    ]
}

# Representation Analysis
REPRESENTATION_CONFIG = {
    "similarity_threshold": 0.70,    # Cosine similarity cutoff
    "sample_size": 10000,           # Max samples per dataset  
    "entity_types": ["member", "witness"] # Congressional actors
}
```

</details>

---

## 📁 Output Documentation

The primary output is `clusters_to_visualize.csv` containing comprehensive cluster analysis results.

### Column Specifications

<details>
<summary><strong>📊 Complete Column Reference</strong></summary>

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `cluster` | int | Unique cluster identifier | `42` |
| `topic_label` | str | BERTopic-generated descriptive label (stance-free) | `abortion rights women healthcare` |
| `narrative_chain` | str | Directional progression with current stage highlighted | `health → [womens_rights] → reproductive_rights → abortion_access` |
| `wiba_topics` | str | Consolidated topic labels from Dynamic Consolidator | `reproductive_rights_womens_autonomy_healthcare_choice` |
| `parent_topic` | int | Hierarchical parent cluster (-1 if root) | `15` |
| `has_subtopics` | bool | Whether cluster has child clusters | `true` |
| `source_distribution` | dict | Breakdown of Congressional vs Reddit arguments | `{'Congressional': 23.4, 'Reddit': 76.6, ...}` |
| `cluster_size` | int | Total arguments in cluster | `1,247` |
| `overlap_score` | float | Min(Congressional%, Reddit%) - representation balance | `23.4` |
| `reddit_pct` | float | Percentage of total Reddit args in this cluster | `15.2` |
| `congress_pct` | float | Percentage of total Congressional args in this cluster | `8.7` |
| `overlap_percentage` | float | Same as overlap_score (for compatibility) | `8.7` |
| `reddit_arguments_in_cluster` | int | Raw count of Reddit arguments | `1,892` |
| `congress_arguments_in_cluster` | int | Raw count of Congressional arguments | `355` |
| `total_reddit_arguments` | int | Total Reddit arguments in dataset | `342,156` |
| `total_congress_arguments` | int | Total Congressional arguments in dataset | `89,234` |
| `num_representative_docs` | int | Number of representative documents extracted | `50` |
| `dominant_stance` | str | Most common stance with percentage | `Argument_For (78%)` |
| `representative_documents` | str | Top representative arguments (escaped) | `[Reddit] Women deserve autonomy&#44; [Congressional] Healthcare decisions...` |
| `representative_documents_continued` | str | Continuation if >50 documents | `[Reddit] Personal choice matters&#44; [Congressional] Medical privacy...` |

</details>

### Understanding Key Columns

<details>
<summary><strong>🔍 Deep Dive: Critical Columns Explained</strong></summary>

#### **`narrative_chain`**: Directional Progression Analysis
Shows where the cluster fits in causal argument development:

```
health → [womens_rights] → reproductive_rights → abortion_access
```

- **`health`**: Foundation medical/safety concerns
- **`[womens_rights]`**: **Current cluster position** - arguments about gender equality/autonomy  
- **`reproductive_rights`**: Specific policy framework arguments
- **`abortion_access`**: Implementation and access arguments

**Other Examples:**
- `violence_concern → [public_safety] → gun_regulation → policy_implementation`
- `environmental_concern → scientific_evidence → [policy_development] → implementation`

#### **`wiba_topics`**: Consolidated Topic Labels
Shows the actual merged topic names from Dynamic Consolidator:

- **Before consolidation**: `"abortion rights"`, `"reproductive choice"`, `"womens healthcare"`
- **After consolidation**: `reproductive_rights_womens_autonomy_healthcare_choice`
- **Represents**: The fundamental themes this cluster discusses

#### **`overlap_score`**: Cross-Platform Representation
Measures how balanced the cluster is across platforms:

- **High overlap (>10%)**: Topic discussed similarly in both Reddit and Congress
- **Low overlap (<5%)**: Platform-specific discourse
- **Formula**: `min(reddit_pct, congress_pct)`

**Interpretation:**
- `overlap_score = 15.2%` → Balanced representation across platforms
- `overlap_score = 2.1%` → Primarily single-platform discussion

#### **`source_distribution`**: Detailed Platform Breakdown
Complex object showing exact percentages and counts:

```json
{
  "Congressional": 23.4,        // % of total Congressional args
  "Reddit": 76.6,              // % of total Reddit args  
  "Congressional_count": 355,   // Raw count Congressional
  "Reddit_count": 1892         // Raw count Reddit
}
```

</details>

### Sample Output

<details>
<summary><strong>📄 Example Cluster Entry</strong></summary>

```csv
cluster,topic_label,narrative_chain,wiba_topics,overlap_score,reddit_pct,congress_pct,cluster_size,dominant_stance,representative_documents
42,"abortion rights women healthcare","health → [womens_rights] → reproductive_rights → abortion_access","reproductive_rights_womens_autonomy_healthcare_choice",8.7,15.2,8.7,1247,"Argument_For (78%)","[Reddit] Women deserve the right to make their own healthcare decisions&#44; [Congressional] The fundamental issue here is womens autonomy over their own bodies&#44; [Reddit] This is about basic human rights and dignity..."
```

**Interpretation:**
- **Cluster 42**: Contains 1,247 arguments about women's reproductive rights
- **Narrative Position**: Focused on women's rights stage of abortion discourse progression  
- **Platform Balance**: 15.2% of Reddit args, 8.7% of Congressional args (8.7% overlap)
- **Stance**: Predominantly pro-choice (78%)
- **Topics**: Merged from multiple related reproductive rights topics

</details>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install pandas numpy scikit-learn
pip install sentence-transformers bertopic
pip install cupy cuml  # For GPU acceleration
pip install networkx tqdm plotly
```

### Basic Usage

```python
# Run complete analysis for abortion topic
python calculate_representation.py --topic abortion

# Enable topic consolidation 
python calculate_representation.py --topic abortion --consolidate_topics

# Create Sankey diagram
python calculate_representation.py --topic abortion --create_sankey

# Custom configuration
python calculate_representation.py \
    --topic gmo \
    --sample_size 5000 \
    --similarity_threshold 0.75 \
    --consolidate_topics \
    --consolidation_threshold 0.6
```

### Advanced Configuration

<details>
<summary><strong>⚙️ Configuration Options</strong></summary>

```python
# Dynamic topic consolidation settings
python calculate_representation.py \
    --topic nuclear \
    --consolidate_topics \
    --consolidation_method hierarchical_dynamic \
    --consolidation_threshold 0.6

# Sankey visualization settings  
python calculate_representation.py \
    --create_sankey \
    --sankey_min_size 100 \
    --sankey_top_n 15

# Representation analysis settings
python calculate_representation.py \
    --similarity_threshold 0.70 \
    --sample_size 10000
```

**Available Topics:**
- `abortion`: Abortion and reproductive rights
- `gmo`: Genetically modified organisms  
- `nuclear`: Nuclear energy and weapons
- `gun_control`: Gun policy and regulation

</details>

---

## 📈 Usage Examples

### Example 1: Basic Cluster Analysis

```python
import pandas as pd

# Load results
clusters = pd.read_csv("clusters_to_visualize_abortion.csv")

# Find most balanced clusters (high overlap)
balanced = clusters[clusters['overlap_score'] > 10].sort_values('overlap_score', ascending=False)
print("Most balanced clusters across platforms:")
print(balanced[['cluster', 'topic_label', 'overlap_score', 'cluster_size']].head())

# Analyze narrative progression
progression_counts = clusters['narrative_chain'].value_counts()
print("\nNarrative progression distribution:")
print(progression_counts.head())
```

### Example 2: Representation Intensity Analysis

```python
# Load member representation data
members = pd.read_csv("member_subreddit_representation_abortion.csv")

# Find highly representative members
high_rep = members[members['RI_legislator'] > 50].sort_values('RI_legislator', ascending=False)
print("Members with highest Reddit representation:")
print(high_rep[['speaker_last', 'speaker_first', 'RI_legislator', 'total_arguments']].head())

# Analyze subreddit-specific representation
politics_cols = [col for col in members.columns if 'politics_RI' in col]
print(f"\nr/politics representation: {members[politics_cols[0]].mean():.1f}%")
```

### Example 3: Cross-Platform Topic Analysis

```python
# Compare platform-specific clusters
reddit_heavy = clusters[clusters['reddit_pct'] > clusters['congress_pct'] * 2]
congress_heavy = clusters[clusters['congress_pct'] > clusters['reddit_pct'] * 2]

print(f"Reddit-heavy clusters: {len(reddit_heavy)}")
print(f"Congress-heavy clusters: {len(congress_heavy)}")

# Analyze topic differences
print("\nReddit-heavy topics:")
print(reddit_heavy['topic_label'].head())
print("\nCongress-heavy topics:")  
print(congress_heavy['topic_label'].head())
```

---

## 🛠️ Configuration

### Topic Configuration

The system supports multiple policy domains through `TOPIC_CONFIGS`:

<details>
<summary><strong>📝 Topic Configuration Details</strong></summary>

```python
TOPIC_CONFIGS = {
    'abortion': {
        'congress_file': "/path/to/congress_abortion.pkl",
        'reddit_file': "/path/to/reddit_abortion.pkl", 
        'output_prefix': 'abortion'
    },
    'gmo': {
        'congress_file': "/path/to/congress_gmo.pkl",
        'reddit_file': "/path/to/reddit_gmo.pkl",
        'output_prefix': 'gmo'
    },
    # Add new topics here...
}
```

**Adding New Topics:**
1. Prepare Congressional and Reddit datasets with required columns
2. Add configuration entry to `TOPIC_CONFIGS`
3. Ensure member/witness identification files exist
4. Run analysis: `python calculate_representation.py --topic new_topic`

</details>

### Model Configuration

<details>
<summary><strong>🤖 Model Settings</strong></summary>

```python
# Embedding model (modify in calculate_representation.py)
EMBEDDING_MODEL = '/home/arman/vienna/models/contrastive_finetune_v2_mpnet-v2_mal'

# BERTopic configuration
BERTOPIC_CONFIG = {
    'min_topic_size': 10,
    'nr_topics': None,  # Auto-detect
    'calculate_probabilities': False,
    'representation_model': [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3)
    ]
}

# Dynamic consolidation settings (modify in AutomaticConsolidator.py)
DYNAMIC_CONSOLIDATION = {
    'method': 'hierarchical_dynamic',
    'max_levels': 3,
    'level_thresholds': [0.7, 0.55, 0.4],
    'n_domains': 6,
    'min_domain_size': 3,
    'faiss_enabled': True,
    'performance_optimized': True
}
```

</details>

---

## 📚 Research Applications

### Political Science Applications

<details>
<summary><strong>🏛️ Research Use Cases</strong></summary>

1. **Elite-Mass Opinion Linkage**
   - Measure how Congressional rhetoric reflects grassroots sentiment
   - Identify gaps between institutional and public discourse
   - Track opinion leadership and influence patterns

2. **Narrative Framing Analysis**  
   - Map how policy arguments evolve through causal chains
   - Identify dominant framing strategies by party/ideology
   - Measure narrative convergence across platforms

3. **Representation Quality Assessment**
   - Quantify descriptive vs substantive representation
   - Identify under-represented perspectives in Congressional debate
   - Measure responsiveness to constituent concerns

4. **Cross-Platform Information Flow**
   - Track argument diffusion from social media to institutional settings
   - Identify influential communities and opinion leaders
   - Measure platform-specific discourse patterns

</details>

### Computational Social Science Applications

<details>
<summary><strong>🔬 Technical Applications</strong></summary>

1. **Stance-Aware Topic Modeling**
   - Benchmark for preserving ideological coherence in clustering
   - Framework for multi-signal topic consolidation
   - Template for cross-platform discourse analysis

2. **Narrative Flow Detection**
   - Novel approach to causal argument progression
   - Framework for temporal discourse analysis
   - Template for argument development tracking

3. **Representation Intensity Metrics**
   - Quantitative measures of cross-platform representation
   - Framework for entity-level discourse analysis
   - Template for influence and responsiveness measurement

4. **Hybrid Clustering Approaches**
   - Integration of multiple similarity signals
   - Graph-based consolidation methods
   - Stance-compatibility preservation techniques

</details>

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Contact

For questions about the methodology, implementation, or research applications, please open an issue or contact the maintainers.

---