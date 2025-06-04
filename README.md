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

The analysis generates several output files:
- `clusters_to_visualize.csv`: Cluster statistics and representative arguments
- `member_subreddit_representation.csv`: Member representation across subreddits
- `witness_subreddit_representation.csv`: Witness representation across subreddits