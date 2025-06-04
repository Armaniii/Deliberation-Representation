# Deliberation Representation

This project analyzes argumentation patterns across Reddit discussions and US Congressional Hearings.

## Usage

Run the analysis for any topic using:

```bash
python calculate_representation.py --topic [abortion|gmo|nuclear|gun_control]
```

### Available Topics
- **abortion**: Abortion-related arguments
- **gmo**: GMO (Genetically Modified Organisms) discussions
- **nuclear**: Nuclear energy/weapons debates
- **gun_control**: Gun control policy arguments

### Optional Parameters
- `--sample_size`: Number of samples for UMAP visualization (default: 500)
- `--similarity_threshold`: Cosine similarity threshold for matching (default: 0.70)

### Example
```bash
# Run analysis for GMO topic with custom parameters
python calculate_representation.py --topic gmo --sample_size 1000 --similarity_threshold 0.75
```

## Features

- **Embedding Generation**: Creates embeddings for text segments using SentenceTransformer
- **UMAP Visualization**: Reduces high-dimensional embeddings to 2D space for visualization
- **Clustering**: Uses HDBSCAN for density-based clustering of arguments
- **Representation Analysis**: Calculates how congressional members/witnesses are represented in Reddit discussions
- **GPU Acceleration**: Utilizes CUDA-enabled GPUs for faster computation

## Output Files

The analysis generates several output files with topic-specific naming (e.g., `_gmo`, `_gun_control`, `_abortion`, `_nuclear`):

### Clustering and Visualization
- **`clusters_to_visualize_{topic}.csv`**: 
  - Cluster statistics from UMAP/HDBSCAN analysis
  - Contains: cluster ID, size, overlap scores, Reddit/Congressional percentages
  - Representative arguments for each cluster (up to 50 per cluster)
  - Source distribution showing Congressional vs Reddit representation

### Representation Analysis
- **`member_subreddit_representation_{topic}.csv`**: 
  - Congressional member Representation Intensity RI(L) calculations
  - Contains: member info (name, govtrack ID, congress), total arguments
  - RI_legislator: Percentage of member's arguments that match Reddit arguments
  - Per-subreddit RI calculations showing alignment with specific communities
  - Follows formula: RI(L) = |A_L(R)| / |A(L)| * 100

- **`witness_subreddit_representation_{topic}.csv`**: 
  - Congressional hearing witness Representation Intensity RI(L) calculations  
  - Contains: witness info (name, hearing file), total arguments
  - RI_legislator: Percentage of witness's arguments that match Reddit arguments
  - Per-subreddit RI calculations for detailed community alignment analysis

- **`subreddit_representation_intensity_{topic}.csv`**: 
  - Subreddit Representation Intensity RI(S) calculations
  - Contains: subreddit name, total arguments, matches to Congress
  - RI_subreddit: Percentage of subreddit arguments represented in Congressional hearings
  - Follows formula: RI(S) = |A_S(C)| / |A(S)| * 100

### Missing Data Tracking
- **`missing_members_{topic}.csv`**: 
  - Members/witnesses with no similarity matches in Reddit data
  - Contains: entity information and reason for missing matches
  - Helps identify gaps in representation analysis

### Data Interpretation

**Similarity Threshold**: Default 0.70 cosine similarity between embeddings
- Higher threshold = more strict matching
- Lower threshold = more lenient matching

**Overlap Score**: Minimum of Congressional and Reddit percentages for each cluster
- Higher scores indicate better cross-platform representation
- Lower scores suggest platform-specific argument patterns

**Representation Intensity Metrics**:
- **RI(L)**: Legislator Representation Intensity - Percentage of legislator's arguments that align with Reddit discourse
- **RI(S)**: Subreddit Representation Intensity - Percentage of subreddit arguments represented in Congressional hearings
- Higher RI values indicate stronger alignment between platforms
- Values range from 0% (no alignment) to 100% (complete alignment)