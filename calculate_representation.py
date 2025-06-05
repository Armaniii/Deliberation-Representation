"""
Calculate representation analysis for congressional and Reddit data across different topics.

This module provides functions to analyze argumentation patterns across Reddit discussions 
and US Congressional Hearings for different topics (abortion, nuclear, gmo, gun control).

Usage:
    python calculate_representation.py --topic [abortion|gmo|nuclear|gun_control]

Examples:
    # Run analysis for abortion topic with default parameters
    python calculate_representation.py --topic abortion
    
    # Run analysis for GMO topic with custom parameters
    python calculate_representation.py --topic gmo --sample_size 1000 --similarity_threshold 0.75

Available topics:
    - abortion: Abortion-related arguments
    - gmo: GMO (Genetically Modified Organisms) discussions  
    - nuclear: Nuclear energy/weapons debates
    - gun_control: Gun control policy arguments
"""

import os
import ast
import csv
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import cupy as cp
from cuml.manifold import UMAP
from sentence_transformers import SentenceTransformer, util
# from sklearn.preprocessing import StandardScaler  # Not needed - sentence transformers already normalized
import plotly.graph_objects as go
import tqdm
from cuml.cluster import HDBSCAN

# BERTopic imports
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

# KeyBERT is integrated into BERTopic via KeyBERTInspired representation

# Import topic consolidation
from AutomaticConsolidator import AutomaticTopicConsolidator


# Configuration for different topics
TOPIC_CONFIGS = {
    'abortion': {
        'congress_file': "/home/arman/nature/congress_data/congress_argument_analysis_results_all.pkl",
        'reddit_file': "/home/arman/nature/reddit_data/reddit_argument_analysis_results_all_strat.pkl",
        'output_prefix': 'abortion'
    },
    'gmo': {
        'congress_file': "/home/arman/nature/congress_data/gmo/congress_argument_analysis_results_all.pkl",
        'reddit_file': "/home/arman/nature/reddit_data/gmo/reddit_argument_analysis_results_all_gmo.pkl",
        'output_prefix': 'gmo'
    },
    'nuclear': {
        'congress_file': "/home/arman/apsa/congress_data/nuclear/hearings_nuclear_stance.csv",
        'reddit_file': "/home/arman/nature/reddit_data/nuclear/reddit_argument_analysis_results_all_nuclear.pkl",
        'output_prefix': 'nuclear'
    },
    'gun_control': {
        'congress_file': "/home/arman/nature/congress_data/gun/congress_argument_analysis_results_all.pkl",
        'reddit_file': "/home/arman/nature/reddit_data/gun/reddit_argument_analysis_results_all_gun.pkl",
        'output_prefix': 'gun_control'
    }
}


def ensure_narrative_coherence(data, min_mixed_size=50):
    """
    Ensure narrative coherence by splitting topics with mixed stances.
    
    Simple and robust approach:
    1. Check each topic for stance diversity
    2. If a topic has multiple stances AND sufficient size, split it
    3. Create topic-stance combinations (e.g., "abortion_pro", "abortion_con")
    
    Parameters:
        data (pd.DataFrame): Data with 'extracted_topic' and 'stance' columns
        min_mixed_size (int): Minimum size for a topic to be split by stance
        
    Returns:
        pd.DataFrame: Data with coherent topics
    """
    
    print("Analyzing topic-stance coherence...")
    
    # Analyze each topic for stance diversity
    topic_stance_analysis = []
    
    for topic in data['extracted_topic'].unique():
        topic_data = data[data['extracted_topic'] == topic]
        stance_counts = topic_data['stance'].value_counts()
        
        analysis = {
            'topic': topic,
            'total_size': len(topic_data),
            'unique_stances': len(stance_counts),
            'stance_distribution': stance_counts.to_dict(),
            'needs_splitting': False
        }
        
        # Decision rule: Split if topic has multiple stances and is large enough
        if (len(stance_counts) > 1 and 
            len(topic_data) >= min_mixed_size and
            all(count >= 10 for count in stance_counts.values)):  # Each stance needs minimum docs
            analysis['needs_splitting'] = True
        
        topic_stance_analysis.append(analysis)
    
    # Apply splitting
    topics_to_split = [a for a in topic_stance_analysis if a['needs_splitting']]
    
    if len(topics_to_split) == 0:
        print("✅ All topics are already narratively coherent!")
        return data
    
    print(f"📝 Splitting {len(topics_to_split)} mixed topics for coherence:")
    
    # Create new coherent topic labels
    data = data.copy()
    data['coherent_topic'] = data['extracted_topic']  # Default: keep original
    
    for analysis in topics_to_split:
        topic = analysis['topic']
        stances = analysis['stance_distribution']
        
        print(f"  • {topic}: {stances}")
        
        # Create topic-stance combinations
        topic_mask = data['extracted_topic'] == topic
        
        for stance in stances.keys():
            if stance in ['NoArgument', 'neutral']:  # Skip non-argumentative stances
                continue
                
            # Create clean stance suffix
            stance_suffix = stance.replace('Argument_', '').replace('_', '').lower()
            new_topic = f"{topic}_{stance_suffix}"
            
            # Update data
            stance_mask = topic_mask & (data['stance'] == stance)
            data.loc[stance_mask, 'coherent_topic'] = new_topic
    
    # Update the extracted_topic column
    data['pre_coherence_topic'] = data['extracted_topic'].copy()
    data['extracted_topic'] = data['coherent_topic']
    
    # Summary
    original_topics = len(data['pre_coherence_topic'].unique())
    coherent_topics = len(data['extracted_topic'].unique())
    
    print(f"✅ Coherence ensured: {original_topics} → {coherent_topics} topics")
    print(f"   Split topics now have consistent narratives within each stance")
    
    return data


def normalize_text(text, source):
    """
    Normalize text based on source to reduce style differences and focus on content.
    
    Parameters:
        text (str): Text to normalize
        source (str): Either 'Reddit' or 'Congressional'
        
    Returns:
        str: Normalized text
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
        
    # Common normalization
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    if source.lower() == 'reddit':
        # Handle informal language
        text = text.lower()  # Lowercase for informal text
        # Remove Reddit-specific markers
        text = re.sub(r'/u/\S+', '', text)  # Remove usernames
        text = re.sub(r'/r/\S+', '', text)  # Remove subreddit references
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)  # Remove deleted markers
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'\\n|\\t', ' ', text)  # Remove escape sequences
        # Remove excessive punctuation
        text = re.sub(r'[!?]{2,}', '.', text)
        text = re.sub(r'\.{3,}', '.', text)
        
    elif source.lower() == 'congressional':
        # Handle formal language
        # Remove formal titles and procedures
        text = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Hon\.|Rep\.|Sen\.)\s*', '', text, flags=re.IGNORECASE)
        # Remove procedural language
        text = re.sub(r'\b(yield.*?time|gentleman.*?yields|gentlewoman.*?yields)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(recognize.*?gentleman|recognize.*?gentlewoman)', '', text, flags=re.IGNORECASE)
        # Remove time stamps and procedural markers
        text = re.sub(r'\[\d+:\d+:\d+\]', '', text)
        text = re.sub(r'\(.*?reserved.*?\)', '', text, flags=re.IGNORECASE)
        # Normalize case - make it lowercase like Reddit for better matching
        text = text.lower()
        
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace again
    text = text.strip()
    
    return text


def parse_embedding(emb):
    """
    Parse string representation of embeddings to numpy array.
    
    Parameters:
        emb: Embedding that could be a string or array
        
    Returns:
        np.array or None: Parsed embedding array or None if parsing fails
    """
    try:
        # If already a numpy array, return as is
        if isinstance(emb, np.ndarray):
            return emb
            
        # Handle empty or null values (check this after numpy array check)
        if emb is None:
            return None
            
        # For pandas scalar values, check if it's NaN
        try:
            if pd.isna(emb):
                return None
        except (TypeError, ValueError):
            # pd.isna might fail on some types, continue processing
            pass
            
        # If it's a list, convert to numpy array
        if isinstance(emb, list):
            return np.array(emb)
            
        # If it's a string, try to parse it
        if isinstance(emb, str):
            # Handle empty strings
            if not emb.strip():
                return None
            # Try parsing as literal
            parsed = ast.literal_eval(emb)
            return np.array(parsed)
            
        # For other types, try direct conversion
        return np.array(emb)
        
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing embedding (type: {type(emb)}): {str(e)[:100]}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing embedding: {str(e)[:100]}")
        return None


def create_embeddings(df):
    """
    Generate embeddings for text segments while preserving DataFrame structure and handling None values.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'text_segment' column
        
    Returns:
        pd.DataFrame: DataFrame with 'embeddings' column, where None text segments map to None embeddings
    """
    model = SentenceTransformer('/home/arman/vienna/models/contrastive_finetune_v2_mpnet-v2_mal')
    # model = SentenceTransformer('/home/arman/vienna/models/contrastive_finetune_v2_nomic_o3',trust_remote_code=True)
    # model = SentenceTransformer('all-mpnet-base-v2')

    print(f"Starting embedding creation for {len(df)} rows")
    
    # Normalize text before embedding if source column exists
    if 'source' in df.columns:
        print("Normalizing text based on source...")
        df['normalized_text'] = df.apply(
            lambda x: normalize_text(x['text_segment'], x['source']), 
            axis=1
        )
        text_column = 'normalized_text'
    else:
        text_column = 'text_segment'
    
    # Create mask for valid text segments
    valid_mask = df[text_column].notna() & (df[text_column].str.strip() != '')
    print(f"Found {valid_mask.sum()} valid text segments out of {len(df)} total")
    
    # Initialize embeddings column with None values (object dtype to store numpy arrays)
    df['embeddings'] = pd.Series([None] * len(df), dtype='object')
    
    if valid_mask.any():
        # Get valid text segments and filter out empty/short ones
        valid_texts = df.loc[valid_mask, text_column].tolist()
        
        # Filter out texts that are too short (less than 3 characters)
        filtered_texts = []
        filtered_indices = []
        for idx, text in zip(df[valid_mask].index, valid_texts):
            if isinstance(text, str) and len(text.strip()) >= 3:
                filtered_texts.append(text.strip())
                filtered_indices.append(idx)
        
        print(f"Processing {len(filtered_texts)} text segments for embedding generation")
        
        if filtered_texts:
            # Generate embeddings only for valid text segments
            valid_embeddings = model.encode(
                filtered_texts,
                show_progress_bar=True,
                batch_size=32
            )
            
            print(f"Generated embeddings shape: {valid_embeddings.shape}")
            
            # Assign embeddings back to original DataFrame
            # Use object dtype to store numpy arrays
            if df['embeddings'].dtype != 'object':
                df['embeddings'] = df['embeddings'].astype('object')
            
            for idx, emb in zip(filtered_indices, valid_embeddings):
                df.at[idx, 'embeddings'] = emb
        
        # Check for any remaining None embeddings
        final_valid_count = df['embeddings'].notna().sum()
        print(f"Final valid embeddings: {final_valid_count}")
        if final_valid_count < len(df):
            print(f"Warning: {len(df) - final_valid_count} rows have invalid/missing embeddings")
    
    return df


def calculate_overlap(source_dict):
    """
    Calculate the overlap percentage between Congressional and Reddit representation.
    
    Parameters:
        source_dict (dict): Dictionary with 'Congressional' and 'Reddit' percentage values
        
    Returns:
        float: Minimum of the two percentages (overlap score)
    """
    congressional_pct = source_dict.get('Congressional', 0)  # Already 0-100
    reddit_pct = source_dict.get('Reddit', 0)  # Already 0-100
    return min(congressional_pct, reddit_pct)  # Return as percentage


def get_representative_arguments(all_data, clusters_to_visualize, n=100):
    """
    Extract the top n most representative arguments for each cluster, ensuring text uniqueness.
    
    Parameters:
        all_data (pd.DataFrame): DataFrame with all data including cluster assignments
        clusters_to_visualize (pd.DataFrame): DataFrame with cluster statistics
        n (int): Number of representative arguments to extract per cluster
        
    Returns:
        pd.DataFrame: DataFrame with representative arguments for each cluster
    """
    representative_args = []
    
    # For each unique cluster
    for cluster_id in clusters_to_visualize['cluster'].unique():
        if cluster_id == -1:  # Skip noise cluster if present
            continue
            
        # Get cluster data
        cluster_data = all_data[all_data['cluster'] == cluster_id]
        
        # Calculate centroid in embedding space
        cluster_embeddings = np.stack(cluster_data['embeddings'].values)
        centroid_embedding = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distance to centroid for each point in embedding space
        cluster_data = cluster_data.copy()  # Fix SettingWithCopyWarning
        cluster_data['distance_to_centroid'] = cluster_data['embeddings'].apply(
            lambda emb: np.linalg.norm(emb - centroid_embedding)
        )
        
        # Sort by distance
        sorted_data = cluster_data.sort_values('distance_to_centroid')
        
        # Deduplicate by text_segment, keeping only the closest instance of each text
        seen_texts = set()
        unique_representative = []
        
        for _, row in sorted_data.iterrows():
            # Use a normalized version of the text for deduplication
            text_normalized = str(row['text_segment']).strip().lower()
            
            if text_normalized not in seen_texts:
                seen_texts.add(text_normalized)
                unique_representative.append(row)
                
            # Stop once we have n unique texts
            if len(unique_representative) >= n:
                break
        
        # Convert back to DataFrame
        if unique_representative:
            representative_df = pd.DataFrame(unique_representative)
            representative_df['representative_rank'] = range(1, len(representative_df) + 1)
            representative_args.append(representative_df)
    
    # Combine all representative arguments
    if representative_args:
        return pd.concat(representative_args, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_subreddit_representation(cluster, subreddit_column, all_data):
    """
    Calculate subreddit representation within a cluster.
    
    Parameters:
        cluster (int): Cluster ID
        subreddit_column (str): Name of the subreddit column
        all_data (pd.DataFrame): DataFrame with all data
        
    Returns:
        dict: Dictionary with subreddit representation percentages
    """
    cluster_data = all_data[all_data['cluster'] == cluster]
    subreddit_counts = cluster_data[subreddit_column].value_counts()
    subreddit_totals = all_data[all_data[subreddit_column].notnull()].groupby(subreddit_column).size()
    subreddit_representation = ((subreddit_counts / subreddit_totals) * 100).to_dict()
    return subreddit_representation


def generate_umap_visualization_balanced(congressional_data, reddit_data, sample_size=10000, 
                                       num_components=2, congress_length=0, reddit_length=0, topic=''):
    """
    Generate a balanced UMAP visualization of congressional and Reddit data using GPU acceleration.
    
    Parameters:
        congressional_data (pd.DataFrame): Congressional hearing data
        reddit_data (pd.DataFrame): Reddit discussion data
        sample_size (int): Maximum number of samples to use from each dataset
        num_components (int): Number of UMAP components (default: 2)
        congress_length (int): Total length of congressional data
        reddit_length (int): Total length of reddit data
        
    Returns:
        None: Saves visualization and cluster statistics to files
    """
    # Parse embeddings
    congressional_data['embeddings'] = congressional_data['embeddings'].apply(parse_embedding)
    reddit_data['embeddings'] = reddit_data['embeddings'].apply(parse_embedding)
    congressional_data['source'] = 'Congressional'
    reddit_data['source'] = 'Reddit'

    # Filter out invalid embeddings
    congressional_data = congressional_data.dropna(subset=['embeddings'])
    reddit_data = reddit_data.dropna(subset=['embeddings'])

    print(f"Congressional columns: {congressional_data.columns.tolist()}")
    print(f"Reddit columns: {reddit_data.columns.tolist()}")

    # Sample data if necessary (use sample_size=0 to use all data)
    if sample_size > 0:
        if len(congressional_data) > sample_size:
            congressional_data = congressional_data.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} congressional arguments from {len(congressional_data)}")
        if len(reddit_data) > sample_size:
            reddit_data = reddit_data.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} reddit arguments from {len(reddit_data)}")
    else:
        print("Using all available data (no sampling)")

    # Combine data
    all_data = pd.concat([congressional_data, reddit_data], ignore_index=True)
    
    # Ensure we have a text column and normalize it
    if 'text_to_embed' not in all_data.columns:
        all_data['text_to_embed'] = all_data['text_segment']
    
    # Apply text normalization
    print("Normalizing text for clustering...")
    all_data['normalized_text'] = all_data.apply(
        lambda x: normalize_text(x['text_to_embed'], x['source']), 
        axis=1
    )
    
    # Consolidate topics before clustering if requested
    consolidation_enabled = globals().get('consolidation_enabled', False)
    consolidation_method = globals().get('consolidation_method', 'hybrid')
    consolidation_threshold = globals().get('consolidation_threshold', 0.7)
    
    if consolidation_enabled:
        print("\n=== CONSOLIDATING FRAGMENTED TOPICS ===")
        consolidator = AutomaticTopicConsolidator(
            # embedding_model=sentence_transformer,
            similarity_threshold=consolidation_threshold
        )
        
        all_data, consolidation_info = consolidator.consolidate_topics(
            all_data, 
            method=consolidation_method
        )
        
        # Update extracted_topic to use consolidated topics
        all_data['original_extracted_topic'] = all_data['extracted_topic'].copy()
        all_data['extracted_topic'] = all_data['consolidated_topic']
        
        print(f"Topic consolidation complete:")
        print(f"- Original topics: {len(all_data['original_extracted_topic'].unique())}")
        print(f"- Consolidated topics: {len(all_data['extracted_topic'].unique())}")
        print(f"- Reduction: {len(all_data['original_extracted_topic'].unique()) - len(all_data['extracted_topic'].unique())} topics merged")
    else:
        print("\n=== SKIPPING TOPIC CONSOLIDATION ===")
        print("Use --consolidate_topics to enable automatic topic consolidation")
    
    # Apply stance-aware topic splitting for narrative coherence
    print("\n=== ENSURING NARRATIVE COHERENCE ===")
    all_data = ensure_narrative_coherence(all_data)
    print(f"Final coherent topics: {len(all_data['extracted_topic'].unique())}")
        # Quick test: Do embeddings separate pro/con?
    from sklearn.metrics.pairwise import cosine_similarity

    # Group by topic and stance
    # for topic in all_data['extracted_topic'].unique():
    #     topic_data = all_data[all_data['extracted_topic'] == topic]
        
    #     if len(topic_data['stance'].unique()) > 1:
    #         pro_embeddings = np.stack(topic_data[topic_data['stance'] == 'Argument_for']['embeddings'].values)
    #         con_embeddings = np.stack(topic_data[topic_data['stance'] == 'Argument_against']['embeddings'].values)
            
    #         # Within-stance similarity
    #         pro_sim = np.mean(cosine_similarity(pro_embeddings))
    #         con_sim = np.mean(cosine_similarity(con_embeddings))
            
    #         # Cross-stance similarity
    #         cross_sim = np.mean(cosine_similarity(pro_embeddings, con_embeddings))
            
    #         print(f"{topic}: Pro-Pro: {pro_sim:.3f}, Con-Con: {con_sim:.3f}, Pro-Con: {cross_sim:.3f}")
    
    # Stack embeddings
    all_embeddings = np.stack(all_data['embeddings'].values)
    
    from bertopic.backend import BaseEmbedder

    class HybridEmbedder(BaseEmbedder):
        def __init__(self, document_embeddings, sentence_model):
            super().__init__()
            self.document_embeddings = document_embeddings
            self.sentence_model = sentence_model
            self.document_index = 0
            
        def embed(self, documents, verbose=False):
            # If embedding the full document set, use pre-computed
            if len(documents) == len(self.document_embeddings):
                return self.document_embeddings
            else:
                # For word embeddings (MMR), use the sentence model
                return self.sentence_model.encode(documents, show_progress_bar=False)

    # Stack your pre-computed embeddings
    all_embeddings = np.stack(all_data['embeddings'].values)

    # Load sentence transformer for word embeddings
    sentence_transformer = SentenceTransformer('/home/arman/vienna/models/contrastive_finetune_v2_mpnet-v2_mal')
    # sentence_transformer = SentenceTransformer('/home/arman/vienna/models/contrastive_finetune_v2_nomic_o3',trust_remote_code=True)

    # sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
    # Create hybrid embedder
    embedding_model = HybridEmbedder(all_embeddings, sentence_transformer)
    # Use BERTopic for clustering
    print("Using BERTopic for theme discovery...")
    
    # Use the sentence transformer directly as embedding model
    # This allows MMR to properly embed both documents and individual words    
    # Chain representations for richer argument descriptions
    representation_model = [
        KeyBERTInspired(),  # Find key argumentative phrases
        MaximalMarginalRelevance(diversity=0.3)  # Reduce redundancy
    ]
    
    # Custom class-based TF-IDF for arguments
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,  # Reduce common argument markers
        bm25_weighting=True  # Better for varying argument lengths
    )

    # Implement guaranteed pure clustering
    print("Using guaranteed pure clustering strategy...")
    
    def guaranteed_pure_clustering(data, embeddings, min_topic_size=10):
        """Guaranteed stance-pure clustering by processing each topic-stance separately."""
        all_results = []
        global_cluster_id = 0
        topic_model_storage = {}
        
        # Process each topic-stance combination independently
        for (topic, stance), group in data.groupby(['extracted_topic', 'stance']):
            if topic == "No Topic" or stance == "NoArgument":
                continue
                
            print(f"Processing: {topic} - {stance} ({len(group)} documents)")
            
            # Get embeddings for this group
            group_indices = group.index.tolist()
            group_embeddings = embeddings[group_indices]
            
            # Handle small groups
            if len(group) < min_topic_size * 2:
                group = group.copy()
                group['cluster'] = global_cluster_id
                group['cluster_label'] = f"{topic}_{stance}_general"
                all_results.append(group)
                global_cluster_id += 1
                continue
            
            # Configure BERTopic for this specific topic-stance
            topic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=min_topic_size,
                nr_topics=None,  # Let it find natural groupings
                calculate_probabilities=False,
                verbose=False
            )
            
            try:
                # Find sub-clusters within this topic-stance
                clusters, _ = topic_model.fit_transform(
                    group['normalized_text'].tolist(),
                    embeddings=group_embeddings
                )
            except (IndexError, ValueError) as e:
                print(f"→ BERTopic failed for {topic}-{stance}: {e}")
                print(f"→ Assigning as single cluster")
                # Fallback: treat entire group as one cluster
                group = group.copy()
                group['cluster'] = global_cluster_id
                group['cluster_label'] = f"{topic}_{stance}_single"
                all_results.append(group)
                global_cluster_id += 1
                continue
            
            # Store topic model for later use
            topic_model_storage[f"{topic}_{stance}"] = topic_model
            
            # Handle outliers and assign global IDs
            cluster_mapping = {}
            for local_cluster in set(clusters):
                if local_cluster == -1:
                    cluster_mapping[local_cluster] = -1
                else:
                    cluster_mapping[local_cluster] = global_cluster_id
                    global_cluster_id += 1
            
            # Apply mapping
            group = group.copy()
            group['cluster'] = [cluster_mapping[c] for c in clusters]
            
            # Create descriptive labels using KeyBERTInspired from BERTopic
            for local_cluster in set(clusters):
                if local_cluster != -1:
                    # Get documents in this cluster
                    cluster_mask = np.array(clusters) == local_cluster
                    cluster_docs = [group['normalized_text'].iloc[i] for i in range(len(clusters)) if cluster_mask[i]]
                    
                    # Use BERTopic's KeyBERTInspired representation for meaningful labels
                    try:
                        # Get the topic representation from BERTopic
                        topic_representation = topic_model.get_topic(local_cluster)
                        
                        if topic_representation and len(topic_representation) > 0:
                            # Extract keywords from BERTopic's KeyBERTInspired representation
                            # topic_representation is a list of (word, score) tuples
                            keywords = []
                            for word, score in topic_representation[:10]:  # Get top 10 terms
                                if (score > 0.01 and  # Minimum relevance score
                                    len(word) > 2 and  # Minimum word length
                                    word.lower() not in topic.lower() and  # Not already in topic
                                    word.lower() not in stance.lower()):   # Not already in stance
                                    keywords.append(word)
                            
                            # Filter out common stop words and stance terms
                            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                                        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 
                                        'been', 'be', 'have', 'has', 'had', 'my', 'your', 'you', 'i', 'we',
                                        'this', 'that', 'these', 'those', 'will', 'would', 'could', 'should'}
                            
                            # Also filter stance-related terms
                            stance_words = {'argument', 'for', 'against', 'pro', 'con', 'support', 'oppose', 
                                          'argumentfor', 'argumentagainst', 'argument_for', 'argument_against'}
                            
                            meaningful_terms = [word for word in keywords 
                                              if (word.lower() not in stop_words and 
                                                  word.lower() not in stance_words)]
                            
                            # Take top 3 meaningful terms
                            clean_label = '_'.join(meaningful_terms[:3]) if meaningful_terms else 'general'
                        else:
                            clean_label = 'general'
                            
                    except Exception as e:
                        print(f"KeyBERTInspired extraction failed: {e}, using fallback")
                        clean_label = 'general'
                    
                    cluster_indices = group.iloc[cluster_mask].index
                    group.loc[cluster_indices, 'cluster_label'] = f"{topic}_{stance}_{clean_label}"
            
            # Handle outliers
            outlier_mask = np.array(clusters) == -1
            if outlier_mask.any():
                outlier_indices = group.iloc[outlier_mask].index
                group.loc[outlier_indices, 'cluster_label'] = f"{topic}_{stance}_outlier"
            
            all_results.append(group)
            
            # Print results
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_outliers = sum(1 for c in clusters if c == -1)
            print(f"→ Found {n_clusters} narrative variants, {n_outliers} outliers")
        
        # Combine results
        result = pd.concat(all_results, ignore_index=True)
        print(f"TOTAL CLUSTERS: {result['cluster'].nunique()}")
        print("ALL CLUSTERS ARE GUARANTEED PURE (single topic, single stance)")
        
        return result, topic_model_storage
    
    # Run guaranteed pure clustering
    all_data, topic_model_storage = guaranteed_pure_clustering(all_data, all_embeddings, min_topic_size=10)
    cluster_labels = all_data['cluster'].values
    
    # Create a combined topic model for export compatibility
    class CombinedTopicModel:
        def __init__(self, storage, all_data):
            self.storage = storage
            self.all_data = all_data
            self.cluster_to_label = {}
            
            # Build mapping from cluster ID to proper BERTopic label
            for cluster_id in all_data['cluster'].unique():
                if cluster_id != -1 and 'cluster_label' in all_data.columns:
                    # Get the cluster label from the data
                    cluster_labels = all_data[all_data['cluster'] == cluster_id]['cluster_label'].dropna()
                    if len(cluster_labels) > 0:
                        self.cluster_to_label[cluster_id] = cluster_labels.iloc[0]
            
        def get_topic_info(self):
            # Create proper topic info with meaningful labels
            topic_data = []
            
            for cluster_id in sorted(self.all_data['cluster'].unique()):
                if cluster_id == -1:
                    continue
                    
                cluster_data = self.all_data[self.all_data['cluster'] == cluster_id]
                count = len(cluster_data)
                
                # Get the label
                if cluster_id in self.cluster_to_label:
                    label = self.cluster_to_label[cluster_id]
                    # Format as BERTopic style: "0_word1_word2_word3"
                    name = f"{cluster_id}_{label}"
                else:
                    name = f"{cluster_id}_cluster"
                
                topic_data.append({
                    'Topic': cluster_id,
                    'Count': count,
                    'Name': name
                })
            
            if topic_data:
                return pd.DataFrame(topic_data)
            else:
                return pd.DataFrame({'Topic': [], 'Count': [], 'Name': []})
    
    topic_model = CombinedTopicModel(topic_model_storage, all_data)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(f"\nBERTopic found {len(topic_info)} topics (excluding outliers)")
    
    # Calculate representative documents for pure clusters
    print("\nCalculating representative documents for pure clusters...")
    documents = all_data['normalized_text'].tolist()
    topics = cluster_labels
    
    # Create DataFrame with all documents and their cluster assignments
    df_docs_topics = pd.DataFrame({
        "Document": documents, 
        "Topic": topics,
        "original_index": all_data.index
    })
    
    # Calculate representative documents for each cluster
    representative_docs = {}
    
    for cluster_id in all_data['cluster'].unique():
        if cluster_id != -1:  # Skip outliers
            # Get all documents in this cluster
            cluster_data = all_data[all_data['cluster'] == cluster_id]
            cluster_docs = cluster_data['normalized_text'].tolist()
            cluster_indices = cluster_data.index.tolist()
            
            if len(cluster_indices) > 0:
                # Get embeddings for documents in this cluster
                cluster_embeddings = all_embeddings[cluster_indices]
                
                # Calculate centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                
                # Get indices sorted by distance (closest first)
                sorted_indices = np.argsort(distances)
                
                # Get most representative documents
                n_docs = min(100, len(sorted_indices))
                sorted_docs = [cluster_docs[i] for i in sorted_indices[:n_docs]]
                
                representative_docs[cluster_id] = sorted_docs
    
    print(f"Representative documents calculated for {len(representative_docs)} clusters")
    
    # Set hierarchical_topics to None since we don't have traditional hierarchical clustering
    hierarchical_topics = None
    
    # Store topic model for later use
    all_data['topic_model'] = topic_model
    
    # Print clustering statistics
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len([c for c in unique_clusters if c != -1])
    n_noise = np.sum(cluster_labels == -1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Cluster distribution: {np.unique(cluster_labels, return_counts=True)}")
    
    # Calculate statistics
    total_congressional = len(congressional_data) if congress_length == 0 else congress_length
    total_reddit = len(reddit_data) if reddit_length == 0 else reddit_length
    
    print(f"Total Congressional: {total_congressional}, Total Reddit: {total_reddit}")
    
    # Calculate cluster statistics
    cluster_sizes = all_data.groupby('cluster').size().reset_index(name='cluster_size')
    
    # Group by clusters and calculate proportions
    cluster_stats = all_data.groupby('cluster').agg({
        'source': lambda x: {
            'Congressional': (x[x == 'Congressional'].count() / total_congressional) * 100,
            'Reddit': (x[x == 'Reddit'].count() / total_reddit) * 100,
            'Congressional_count': x[x == 'Congressional'].count(),
            'Reddit_count': x[x == 'Reddit'].count()
        },
        'extracted_topic': lambda x: x.value_counts().idxmax() if 'extracted_topic' in all_data.columns and len(x) > 0 else 'N/A',
        'stance': lambda x: x.value_counts().idxmax() if 'stance' in all_data.columns and len(x) > 0 else 'N/A'
    }).reset_index()
    
    # Rename columns
    cluster_stats.columns = ['cluster', 'source', 'extracted_topic', 'stance']
    
    # Merge with cluster sizes and remove noise cluster
    cluster_stats = cluster_stats.merge(cluster_sizes, on='cluster')
    cluster_stats = cluster_stats[cluster_stats['cluster'] != -1]
    
    # Calculate subreddit representation if available
    if 'subreddit' in all_data.columns:
        cluster_stats['subreddit_representation'] = cluster_stats['cluster'].apply(
            lambda x: calculate_subreddit_representation(x, 'subreddit', all_data))
    
    # Calculate overlap score
    cluster_stats['overlap_score'] = cluster_stats['source'].apply(lambda x: calculate_overlap(x))
    
    # Export cluster statistics and representative documents from BERTopic
    topic_prefix = f'_{topic}' if topic else ''
    export_cluster_data(cluster_stats, all_data, representative_docs, topic_model, hierarchical_topics, df_docs_topics, topic_prefix)
    
    print("UMAP visualization and clustering complete.")
    
    # Optionally create and save visualization figure
    # fig = create_visualization_figure(all_data, cluster_stats)
    # fig.write_image("/home/arman/nature/umap_visualization.png")


def export_cluster_data(cluster_stats, all_data, representative_docs, topic_model, hierarchical_topics, df_docs_topics, topic_prefix=''):
    """
    Export cluster statistics and representative documents from BERTopic to CSV.
    
    Parameters:
        cluster_stats (pd.DataFrame): Cluster statistics
        all_data (pd.DataFrame): All data with cluster assignments
        representative_docs (dict): BERTopic's representative documents per topic
        topic_model: BERTopic model instance
        hierarchical_topics: BERTopic's hierarchical topic structure
        df_docs_topics (pd.DataFrame): Document-topic mapping DataFrame
        topic_prefix (str): Topic prefix for output file naming (e.g., '_gmo', '_gun')
    """
    # Get representative documents for each cluster
    representative_texts = {}
    
    # Create a mapping from normalized text to original data for faster lookup
    text_to_data = {}
    for idx, row in all_data.iterrows():
        text_to_data[row['normalized_text']] = {
            'source': row['source'],
            'text_segment': row['text_segment'],
            'index': idx
        }
    
    for topic_id in representative_docs:
        if topic_id != -1:  # Skip outlier topic
            # BERTopic returns actual document texts (normalized text we passed to it)
            representative_doc_texts = representative_docs[topic_id]
            
            # Get the source and original text for each representative document
            # Process ALL available representative docs (no artificial limit)
            topic_texts = []
            for doc_text in representative_doc_texts:  # Use all available documents
                if doc_text in text_to_data:
                    data = text_to_data[doc_text]
                    source = data['source']
                    original_text = data['text_segment']
                    text_with_source = f"[{source}] {original_text}"
                    topic_texts.append(text_with_source)
                else:
                    # Fallback: use the document text as-is
                    topic_texts.append(f"[Unknown] {doc_text}")
            
            # Join texts with delimiter, properly escaping all problematic chars
            representative_texts[topic_id] = '||'.join(
                str(text).replace('"', '&#34;').replace("'", '&#39;').replace(',', '&#44;').replace('\n', '&#10;').replace('\r', '&#13;').replace('||', '&#124;&#124;') 
                for text in topic_texts
            )
    
    # Convert to DataFrame format with split logic for long lists
    grouped_args_data = []
    
    for cluster, texts in representative_texts.items():
        # Count documents in this cluster
        doc_count = texts.count('||') + 1 if texts else 0
        
        # If more than 50 documents, split into two columns
        if doc_count > 50:
            # Split the text at roughly the middle
            all_docs = texts.split('||')
            mid_point = len(all_docs) // 2
            
            # First half
            docs_part1 = '||'.join(all_docs[:mid_point])
            # Second half
            docs_part2 = '||'.join(all_docs[mid_point:])
            
            grouped_args_data.append({
                'cluster': cluster, 
                'text_segment': docs_part1,
                'text_segment_part2': docs_part2
            })
        else:
            grouped_args_data.append({
                'cluster': cluster, 
                'text_segment': texts,
                'text_segment_part2': ''  # Empty for consistency
            })
    
    grouped_args = pd.DataFrame(grouped_args_data)
    
    # Create export dataframe
    export_data = pd.DataFrame()
    export_data['cluster'] = cluster_stats['cluster']
    
    # Add topic labels from BERTopic (remove stance information)
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        if row['Topic'] != -1:  # Skip outliers
            # Extract meaningful label from the Name field
            # Name format is like "0_abortion_Argument_For_rights_women_choice"
            name_parts = row['Name'].split('_')
            if len(name_parts) > 1:
                # Remove the cluster number prefix
                label_parts = name_parts[1:]
                
                # Remove stance information (Argument_For, Argument_Against, etc.)
                cleaned_parts = []
                for part in label_parts:
                    # Skip stance-related terms (case insensitive)
                    part_lower = part.lower()
                    if part_lower not in ['argument', 'for', 'against', 'argument_for', 'argument_against', 
                                         'pro', 'con', 'support', 'oppose', 'argumentfor', 'argumentagainst']:
                        cleaned_parts.append(part)
                
                # Join with spaces for readability
                if cleaned_parts:
                    topic_labels[row['Topic']] = ' '.join(cleaned_parts[:6])  # Use up to 6 words
                else:
                    # Fallback if all parts were stance-related
                    topic_labels[row['Topic']] = 'general_topic'
            else:
                topic_labels[row['Topic']] = row['Name']
    
    export_data['topic_label'] = export_data['cluster'].map(topic_labels).fillna('Outliers')
    
    # Add narrative chain information with directional flow
    if 'narrative_chain' in all_data.columns:
        narrative_chains = {}
        for cluster_id in export_data['cluster'].unique():
            if cluster_id != -1:
                cluster_data = all_data[all_data['cluster'] == cluster_id]
                # Get the most common narrative chain in this cluster
                chain_counts = cluster_data['narrative_chain'].value_counts()
                if len(chain_counts) > 0:
                    dominant_chain = chain_counts.index[0]
                    
                    # Prioritize directional chains over standalone
                    if dominant_chain != 'standalone':
                        narrative_chains[cluster_id] = dominant_chain
                    else:
                        # If dominant is standalone, check for any directional chains
                        directional_chains = [chain for chain in chain_counts.index 
                                           if chain != 'standalone' and '→' in chain]
                        if directional_chains:
                            # Use the most frequent directional chain
                            for chain in chain_counts.index:
                                if chain in directional_chains:
                                    narrative_chains[cluster_id] = chain
                                    break
                        else:
                            narrative_chains[cluster_id] = 'standalone'
                else:
                    narrative_chains[cluster_id] = 'standalone'
        
        export_data['narrative_chain'] = export_data['cluster'].map(narrative_chains).fillna('standalone')
    
    # Add WIBA topics column - top 4 most frequent topics per cluster
    wiba_topics = {}
    if 'extracted_topic' in all_data.columns:
        for cluster_id in export_data['cluster'].unique():
            if cluster_id != -1:  # Skip outliers
                cluster_data = all_data[all_data['cluster'] == cluster_id]
                if 'extracted_topic' in cluster_data.columns and len(cluster_data) > 0:
                    # Get top 4 most frequent topics in this cluster
                    topic_counts = cluster_data['extracted_topic'].value_counts().head(4)
                    if len(topic_counts) > 0:
                        # Format like BERTopic: topic1_topic2_topic3_topic4
                        top_topics = topic_counts.index.tolist()
                        # Clean and format topics
                        clean_topics = []
                        for topic in top_topics:
                            if pd.notna(topic) and str(topic).strip() != '':
                                clean_topics.append(str(topic).strip().replace(' ', '_'))
                        
                        if clean_topics:
                            wiba_topics[cluster_id] = '_'.join(clean_topics)
                        else:
                            wiba_topics[cluster_id] = 'no_topics'
                    else:
                        wiba_topics[cluster_id] = 'no_topics'
                else:
                    wiba_topics[cluster_id] = 'no_extracted_topics'
            else:
                wiba_topics[cluster_id] = 'outliers'
    
    export_data['wiba_topics'] = export_data['cluster'].map(wiba_topics).fillna('no_wiba_data')
    
    # Add hierarchical information
    hierarchy_info = {}
    if hierarchical_topics is not None and len(hierarchical_topics) > 0:
        # Create a mapping of child topics to their parents
        for _, row in hierarchical_topics.iterrows():
            child_topic = row['Child_Left_ID'] if 'Child_Left_ID' in row else row.get('Topics', [])[0] if 'Topics' in row and len(row['Topics']) > 0 else None
            parent_topic = row['Parent_ID'] if 'Parent_ID' in row else None
            
            if child_topic is not None and parent_topic is not None:
                hierarchy_info[child_topic] = parent_topic
    
    export_data['parent_topic'] = export_data['cluster'].map(hierarchy_info).fillna(-1)
    export_data['has_subtopics'] = export_data['cluster'].apply(
        lambda x: len([k for k, v in hierarchy_info.items() if v == x]) > 0
    )
    
    export_data['source_distribution'] = cluster_stats['source']
    export_data['cluster_size'] = cluster_stats['cluster_size']
    export_data['overlap_score'] = cluster_stats['overlap_score']
    
    # Calculate percentages
    export_data['reddit_pct'] = cluster_stats['source'].apply(lambda x: x.get('Reddit', 0)).round(2)
    export_data['congress_pct'] = cluster_stats['source'].apply(lambda x: x.get('Congressional', 0)).round(2)
    export_data['overlap_percentage'] = export_data.apply(
        lambda row: min(row['reddit_pct'], row['congress_pct']), axis=1).round(2)
    
    # Add raw counts for each cluster
    export_data['reddit_arguments_in_cluster'] = cluster_stats['source'].apply(lambda x: x.get('Reddit_count', 0))
    export_data['congress_arguments_in_cluster'] = cluster_stats['source'].apply(lambda x: x.get('Congressional_count', 0))
    
    # Add total counts (same for all rows but useful for reference)
    total_reddit = cluster_stats['source'].apply(lambda x: x.get('Reddit_count', 0)).sum()
    total_congress = cluster_stats['source'].apply(lambda x: x.get('Congressional_count', 0)).sum()
    export_data['total_reddit_arguments'] = total_reddit
    export_data['total_congress_arguments'] = total_congress
    
    # Add number of representative documents
    export_data['num_representative_docs'] = export_data['cluster'].apply(
        lambda x: len(representative_docs.get(x, [])) if x in representative_docs else 0
    )
    
    # Add stance distribution for each cluster
    if 'stance' in all_data.columns:
        stance_info = {}
        for cluster_id in export_data['cluster'].unique():
            if cluster_id != -1:
                cluster_data = all_data[all_data['cluster'] == cluster_id]
                stance_dist = cluster_data['stance'].value_counts()
                if len(stance_dist) > 0:
                    # Get dominant stance and its percentage
                    dominant_stance = stance_dist.index[0]
                    dominant_pct = (stance_dist.iloc[0] / len(cluster_data)) * 100
                    stance_info[cluster_id] = f"{dominant_stance} ({dominant_pct:.0f}%)"
        
        export_data['dominant_stance'] = export_data['cluster'].map(stance_info).fillna('mixed')
    
    # Merge with representative arguments
    export_data = export_data.merge(grouped_args, on='cluster', how='left')
    
    # Rename the text columns to be more descriptive
    if 'text_segment' in export_data.columns:
        export_data = export_data.rename(columns={
            'text_segment': 'representative_documents',
            'text_segment_part2': 'representative_documents_continued'
        })
        
    # Remove empty continuation column if all values are empty
    if 'representative_documents_continued' in export_data.columns:
        if export_data['representative_documents_continued'].str.strip().eq('').all():
            export_data = export_data.drop('representative_documents_continued', axis=1)
    
    # Save to CSV with topic prefix
    output_file = f"/home/arman/nature/clusters_to_visualize{topic_prefix}.csv"
    export_data.to_csv(output_file, 
                      index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"Cluster data with BERTopic representative documents exported to {output_file}")
    
    # Print brief summary instead of all cluster names
    print(f"\nExported {len(export_data)} clusters with BERTopic labels")
    
    # Save hierarchical topic structure separately
    if hierarchical_topics is not None and len(hierarchical_topics) > 0:
        hierarchy_file = f"/home/arman/nature/topic_hierarchy{topic_prefix}.csv"
        hierarchical_topics.to_csv(hierarchy_file, index=False)
        print(f"Hierarchical topic structure exported to {hierarchy_file}")
    
    # Save enhanced topic information
    enhanced_topic_info = topic_info.copy()
    if len(hierarchy_info) > 0:
        enhanced_topic_info['parent_topic'] = enhanced_topic_info['Topic'].map(hierarchy_info).fillna(-1)
        enhanced_topic_info['has_subtopics'] = enhanced_topic_info['Topic'].apply(
            lambda x: len([k for k, v in hierarchy_info.items() if v == x]) > 0
        )
    
    topic_info_file = f"/home/arman/nature/enhanced_topic_info{topic_prefix}.csv"
    enhanced_topic_info.to_csv(topic_info_file, index=False)
    print(f"Enhanced topic information exported to {topic_info_file}")
    
    # Save complete document-topic mapping with metadata
    # This allows getting ALL documents per topic for future analysis
    if df_docs_topics is not None:
        # Add source information
        df_docs_topics['source'] = all_data['source'].values
        df_docs_topics['text_segment'] = all_data['text_segment'].values
        if 'extracted_topic' in all_data.columns:
            df_docs_topics['extracted_topic'] = all_data['extracted_topic'].values
        if 'stance' in all_data.columns:
            df_docs_topics['stance'] = all_data['stance'].values
        
        docs_topics_file = f"/home/arman/nature/all_documents_topics{topic_prefix}.csv"
        df_docs_topics.to_csv(docs_topics_file, index=False)
        print(f"Complete document-topic mapping exported to {docs_topics_file}")
        print(f"This file contains ALL {len(df_docs_topics)} documents with their topic assignments")


def calculate_representation(data, reddit_data, entity_type='member', similarity_threshold=0.70, topic=''):
    """
    Calculate representation intensity metrics following the formal definitions:
    - RI(S): Percentage of subreddit arguments that match congressional arguments
    - RI(L): Percentage of legislator arguments that match reddit arguments
    
    Parameters:
        data (pd.DataFrame): Congressional data with is_member and is_witness flags
        reddit_data (pd.DataFrame): Reddit data with subreddit information
        entity_type (str): Type of entity ('member' or 'witness')
        similarity_threshold (float): Cosine similarity threshold for matching
        topic (str): Topic for file naming
        
    Returns:
        pd.DataFrame: Statistics for each entity showing representation intensity
    """
    # Determine which flag to use based on entity_type
    if entity_type.lower() == 'member':
        flag_column = 'is_member'
        entity_filter = data[data[flag_column] == 1]
        groupby_columns = ['speaker_last', 'speaker_first', 'govtrack', 'congress']
        topic_prefix = f'_{topic}' if topic else ''
        output_path = f'/home/arman/nature/member_subreddit_representation{topic_prefix}.csv'
    elif entity_type.lower() == 'witness':
        flag_column = 'is_witness'
        entity_filter = data[data[flag_column] == 1]
        groupby_columns = ['speaker_last', 'file_name']
        topic_prefix = f'_{topic}' if topic else ''
        output_path = f'/home/arman/nature/witness_subreddit_representation{topic_prefix}.csv'
    else:
        raise ValueError("entity_type must be either 'member' or 'witness'")
    
    print(f"Processing {entity_type}s: {len(entity_filter)} arguments found")
    
    # Parse embeddings and filter out invalid data
    entity_filter = entity_filter.copy()  # Fix SettingWithCopyWarning
    reddit_data = reddit_data.copy()  # Fix SettingWithCopyWarning
    entity_filter['embeddings'] = entity_filter['embeddings'].apply(parse_embedding)
    reddit_data['embeddings'] = reddit_data['embeddings'].apply(parse_embedding)
    entity_filter = entity_filter.dropna(subset=['embeddings'])
    reddit_data = reddit_data.dropna(subset=['embeddings'])
    
    print(f"After filtering invalid embeddings: {len(entity_filter)} arguments")
    
    # Calculate RI(L): Representation Intensity for Legislators/Witnesses
    # RI(L) = |A_L(R)| / |A(L)| * 100
    # Where |A_L(R)| = number of legislator arguments that match Reddit
    # Where |A(L)| = total number of legislator arguments
    
    print("Calculating RI(L): Legislator Representation Intensity...")
    
    # Initialize results
    legislator_results = []
    missing_members = []
    
    # Process each entity (legislator/witness)
    for entity_key, entity_group in tqdm.tqdm(entity_filter.groupby(groupby_columns), 
                                           desc=f'Processing {entity_type}s'):
        try:
            # Stack embeddings for this entity
            if len(entity_group) == 1:
                entity_embeddings = np.array([entity_group['embeddings'].values[0]])
            else:
                entity_embeddings = np.stack(entity_group['embeddings'].values)
                
            total_entity_args = entity_embeddings.shape[0]  # |A(L)|
            
            # Create initial stats dictionary
            if entity_type.lower() == 'member':
                entity_stats = {
                    'speaker_last': entity_key[0],
                    'speaker_first': entity_key[1],
                    'govtrack': entity_key[2],
                    'congress': entity_key[3],
                    'total_arguments': total_entity_args
                }
            else:  # witness
                entity_stats = {
                    'speaker_last': entity_key[0],
                    'file_name': entity_key[1],
                    'total_arguments': total_entity_args
                }
            
            # Stack all Reddit embeddings for global matching
            all_reddit_embeddings = np.stack(reddit_data['embeddings'].values)
            
            # Calculate similarity between this entity's arguments and all Reddit arguments
            similarity_matrix = util.pytorch_cos_sim(entity_embeddings, all_reddit_embeddings).numpy()
            
            # Find which entity arguments have matches in Reddit (above threshold)
            max_similarities = np.max(similarity_matrix, axis=1)
            entity_args_matched_to_reddit = np.sum(max_similarities > similarity_threshold)  # |A_L(R)|
            
            # Calculate RI(L) = |A_L(R)| / |A(L)| * 100
            ri_legislator = (entity_args_matched_to_reddit / total_entity_args) * 100
            entity_stats['RI_legislator'] = ri_legislator
            entity_stats['arguments_matched_to_reddit'] = entity_args_matched_to_reddit
            
            # Calculate RI(L) per subreddit for detailed analysis
            for subreddit in reddit_data['subreddit'].unique():
                subreddit_group = reddit_data[reddit_data['subreddit'] == subreddit]
                subreddit_embeddings = np.stack(subreddit_group['embeddings'].values)
                
                # Calculate similarity with this specific subreddit
                subreddit_similarity = util.pytorch_cos_sim(entity_embeddings, subreddit_embeddings).numpy()
                subreddit_max_similarities = np.max(subreddit_similarity, axis=1)
                entity_args_matched_to_subreddit = np.sum(subreddit_max_similarities > similarity_threshold)
                
                # RI(L) for this specific subreddit
                ri_legislator_subreddit = (entity_args_matched_to_subreddit / total_entity_args) * 100
                
                entity_stats[f'{subreddit}_RI_legislator'] = ri_legislator_subreddit
                entity_stats[f'{subreddit}_arguments_matched'] = entity_args_matched_to_subreddit
            
            # Track members with no matches
            if entity_args_matched_to_reddit == 0:
                missing_member_info = entity_stats.copy()
                missing_member_info['reason'] = 'No similarity matches found in Reddit'
                missing_members.append(missing_member_info)
            
            legislator_results.append(entity_stats)
            
        except Exception as e:
            print(f"Error processing {entity_key}: {e}")
            continue
    
    # Now calculate RI(S): Subreddit Representation Intensity
    # RI(S) = |A_S(C)| / |A(S)| * 100  
    # Where |A_S(C)| = number of subreddit arguments that match Congressional arguments
    # Where |A(S)| = total number of arguments in that subreddit
    
    print("Calculating RI(S): Subreddit Representation Intensity...")
    
    # Stack all congressional embeddings for global matching
    all_congressional_embeddings = np.stack(entity_filter['embeddings'].values)
    
    subreddit_results = []
    for subreddit in reddit_data['subreddit'].unique():
        subreddit_group = reddit_data[reddit_data['subreddit'] == subreddit]
        total_subreddit_args = len(subreddit_group)  # |A(S)|
        
        if total_subreddit_args == 0:
            continue
            
        subreddit_embeddings = np.stack(subreddit_group['embeddings'].values)
        
        # Calculate similarity between subreddit arguments and all congressional arguments
        similarity_matrix = util.pytorch_cos_sim(subreddit_embeddings, all_congressional_embeddings).numpy()
        
        # Find which subreddit arguments have matches in Congress (above threshold)
        max_similarities = np.max(similarity_matrix, axis=1)
        subreddit_args_matched_to_congress = np.sum(max_similarities > similarity_threshold)  # |A_S(C)|
        
        # Calculate RI(S) = |A_S(C)| / |A(S)| * 100
        ri_subreddit = (subreddit_args_matched_to_congress / total_subreddit_args) * 100
        
        subreddit_results.append({
            'subreddit': subreddit,
            'total_arguments': total_subreddit_args,
            'arguments_matched_to_congress': subreddit_args_matched_to_congress,
            'RI_subreddit': ri_subreddit
        })
    
    # Save subreddit representation intensity results
    if subreddit_results:
        subreddit_df = pd.DataFrame(subreddit_results)
        subreddit_output_path = f'/home/arman/nature/subreddit_representation_intensity{topic_prefix}.csv'
        subreddit_df.to_csv(subreddit_output_path, index=False)
        print(f"Subreddit RI results saved to {subreddit_output_path}")
    
    results = legislator_results
    
    # Convert to DataFrame and save
    entity_stats_df = pd.DataFrame(results)
    
    if len(entity_stats_df) > 0:
        entity_stats_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results to save")
    
    # Save missing members data
    if missing_members:
        missing_df = pd.DataFrame(missing_members)
        missing_output_path = f'/home/arman/nature/missing_members{topic_prefix}.csv'
        missing_df.to_csv(missing_output_path, index=False)
        print(f"Missing members data saved to {missing_output_path}")
        print(f"Found {len(missing_members)} members/witnesses with no matches")
    else:
        print("All members/witnesses had at least one match")
    
    return entity_stats_df


def load_and_preprocess_data(topic='abortion'):
    """
    Load and preprocess congressional and reddit data for a specific topic.
    
    Parameters:
        topic (str): Topic to process ('abortion', 'gmo', 'nuclear', 'gun_control')
        
    Returns:
        tuple: (congressional_data, reddit_data) DataFrames
    """
    config = TOPIC_CONFIGS.get(topic)
    if not config:
        raise ValueError(f"Unknown topic: {topic}. Choose from: {list(TOPIC_CONFIGS.keys())}")
    
    print(f"Loading data for topic: {topic}")
    
    # Load congressional data
    if config['congress_file'].endswith('.pkl'):
        congressional_data = pd.read_pickle(config['congress_file'])
    else:
        congressional_data = pd.read_csv(config['congress_file'])
    
    # Ensure text_segment column exists
    if "text_segment" not in congressional_data.columns:
        congressional_data["text_segment"] = congressional_data.get("processed_text", 
                                                                  congressional_data.get("text", ""))
    
    # Add source column
    congressional_data['source'] = 'Congressional'
    
    # Filter for arguments
    if "argument_prediction" in congressional_data.columns:
        congressional_data = congressional_data[congressional_data['argument_prediction'] == "Argument"]
        congressional_data = congressional_data[congressional_data['extracted_topic'] != "No Topic"]
        congressional_data = congressional_data[congressional_data['stance'] != "NoArgument"]
    elif "label" in congressional_data.columns:
        congressional_data["argument_prediction"] = congressional_data["label"]
        congressional_data = congressional_data[congressional_data['argument_prediction'] == "Argument"]
        congressional_data = congressional_data[congressional_data['extracted_topic'] != "No Topic"]
        congressional_data = congressional_data[congressional_data['stance'] != "NoArgument"]
    
    print(f"Congressional arguments: {len(congressional_data)}")
    
    # Create embeddings with normalization
    congressional_data = create_embeddings(congressional_data)

    # Load reddit data
    reddit_data = pd.read_pickle(config['reddit_file'])
    
    # Ensure text_segment column exists
    if "text_segment" not in reddit_data.columns:
        reddit_data["text_segment"] = reddit_data.get("processed_text", 
                                                     reddit_data.get("text", ""))
    
    # Add source column
    reddit_data['source'] = 'Reddit'
    
    # Filter for arguments
    if "argument_prediction" in reddit_data.columns:
        reddit_data = reddit_data[reddit_data['argument_prediction'] == "Argument"]
        reddit_data = reddit_data[reddit_data['extracted_topic'] != "No Topic"]
        reddit_data = reddit_data[reddit_data['stance'] != "NoArgument"]
    elif "argument" in reddit_data.columns:
        reddit_data["argument_prediction"] = reddit_data["argument"]
        reddit_data = reddit_data[reddit_data['argument_prediction'] == "Argument"]
        reddit_data = reddit_data[reddit_data['extracted_topic'] != "No Topic"]
        reddit_data = reddit_data[reddit_data['stance'] != "NoArgument"]
    
    print(f"Reddit arguments: {len(reddit_data)}")
    
    # Create embeddings with normalization
    reddit_data = create_embeddings(reddit_data)
    
    # Process member and witness information for congressional data
    congressional_data = process_member_witness_data(congressional_data, topic)
    
    return congressional_data, reddit_data


def process_member_witness_data(congressional_data, topic='abortion'):
    """
    Process and add member/witness flags to congressional data.
    
    Parameters:
        congressional_data (pd.DataFrame): Congressional hearing data
        topic (str): Topic to determine which member/witness files to use
        
    Returns:
        pd.DataFrame: Congressional data with is_member and is_witness flags
    """
    # Create bioname
    if 'speaker_first' in congressional_data.columns and 'speaker_last' in congressional_data.columns:
        congressional_data['bioname'] = (congressional_data['speaker_first'].fillna('') + ' ' + 
                                        congressional_data['speaker_last'].fillna(''))
    
    # Load member data based on topic
    if topic == 'gmo':
        member_file = '/home/arman/nature/congress_data/gmo/df_member.csv'
    else:
        member_file = '/home/arman/apsa/congress_data/abortion/df_member_abortion.csv'
    
    if os.path.exists(member_file):
        print(f"Loading member data from: {member_file}")
        df_members = pd.read_csv(member_file)
        print(f"Found {len(df_members)} member records")
        
        # Check available columns for merging
        print(f"Member file columns: {df_members.columns.tolist()}")
        
        # Try different merge strategies based on available columns
        if 'congress' in df_members.columns and 'govtrack' in df_members.columns:
            df_members = df_members.drop_duplicates(subset=['congress', 'govtrack'])
            merge_cols = ['congress', 'govtrack']
        elif 'govtrack' in df_members.columns:
            df_members = df_members.drop_duplicates(subset=['govtrack'])
            merge_cols = ['govtrack']
        else:
            print("Warning: Cannot find suitable merge columns in member file")
            congressional_data['is_member'] = 0
            merge_cols = None
        
        if merge_cols:
            # Merge with members
            before_merge = len(congressional_data)
            congressional_data = pd.merge(
                congressional_data, 
                df_members, 
                on=merge_cols, 
                how='left',
                suffixes=('', '_member')
            )
            print(f"Merge completed: {before_merge} -> {len(congressional_data)} rows")
            
            # Check for successful matches
            member_match_col = 'bioname_member' if 'bioname_member' in congressional_data.columns else None
            if not member_match_col and 'speaker_last_member' in congressional_data.columns:
                member_match_col = 'speaker_last_member'
            
            if member_match_col:
                congressional_data['is_member'] = np.where(congressional_data[member_match_col].notna(), 1, 0)
            else:
                # Try to match based on any _member column
                member_cols = [col for col in congressional_data.columns if col.endswith('_member')]
                if member_cols:
                    congressional_data['is_member'] = np.where(congressional_data[member_cols[0]].notna(), 1, 0)
                else:
                    congressional_data['is_member'] = 0
    else:
        print(f"Member file not found: {member_file}")
        congressional_data['is_member'] = 0
    
    # Load witness data based on topic
    if topic == 'gmo':
        witness_file = '/home/arman/nature/congress_data/gmo/df_witness.csv'
    else:
        witness_file = '/home/arman/apsa/congress_data/abortion/df_witness_abortion.csv'
    
    if os.path.exists(witness_file):
        print(f"Loading witness data from: {witness_file}")
        df_witness = pd.read_csv(witness_file)
        print(f"Found {len(df_witness)} witness records")
        print(f"Witness file columns: {df_witness.columns.tolist()}")
        
        # Handle different column names in witness files
        if 'last_name' in df_witness.columns:
            df_witness.rename(columns={'last_name': 'speaker_last'}, inplace=True)
        
        if 'speaker_last' in df_witness.columns:
            df_witness['speaker_last'] = df_witness['speaker_last'].str.capitalize()
            
        # Try different merge strategies
        if 'file_name' in df_witness.columns and 'speaker_last' in df_witness.columns:
            df_witness = df_witness.drop_duplicates(subset=['file_name', 'speaker_last'])
            merge_cols = ['file_name', 'speaker_last']
        elif 'speaker_last' in df_witness.columns:
            df_witness = df_witness.drop_duplicates(subset=['speaker_last'])
            merge_cols = ['speaker_last']
        else:
            print("Warning: Cannot find suitable merge columns in witness file")
            congressional_data['is_witness'] = 0
            merge_cols = None
        
        if merge_cols:
            # Merge with witnesses
            before_merge = len(congressional_data)
            congressional_data = pd.merge(
                congressional_data, 
                df_witness, 
                on=merge_cols, 
                how='left',
                suffixes=('', '_witness')
            )
            print(f"Witness merge completed: {before_merge} -> {len(congressional_data)} rows")
            
            # Check for successful matches
            witness_match_col = 'witness_info' if 'witness_info' in congressional_data.columns else None
            if not witness_match_col:
                # Try to find any _witness column
                witness_cols = [col for col in congressional_data.columns if col.endswith('_witness')]
                if witness_cols:
                    witness_match_col = witness_cols[0]
            
            if witness_match_col:
                congressional_data['is_witness'] = np.where(congressional_data[witness_match_col].notna(), 1, 0)
            else:
                congressional_data['is_witness'] = 0
    else:
        print(f"Witness file not found: {witness_file}")
        congressional_data['is_witness'] = 0
    
    # Assign member type
    congressional_data['Type'] = np.where(congressional_data.get('witness_info', pd.Series()).notna(), 
                                         'Witness', 'Member')
    
    # Print final statistics
    member_count = congressional_data['is_member'].sum()
    witness_count = congressional_data['is_witness'].sum()
    total_rows = len(congressional_data)
    
    print(f"\\n=== FINAL MEMBER/WITNESS IDENTIFICATION ===")
    print(f"Total congressional rows: {total_rows}")
    print(f"Members identified: {member_count} ({member_count/total_rows*100:.1f}%)")
    print(f"Witnesses identified: {witness_count} ({witness_count/total_rows*100:.1f}%)")
    print(f"Neither member nor witness: {total_rows - member_count - witness_count}")
    
    # Show some examples of identified members and witnesses
    if member_count > 0:
        print(f"\\nExample members:")
        member_examples = congressional_data[congressional_data['is_member'] == 1][['speaker_first', 'speaker_last', 'govtrack']].drop_duplicates().head(5)
        print(member_examples.to_string(index=False))
    
    if witness_count > 0:
        print(f"\\nExample witnesses:")
        witness_examples = congressional_data[congressional_data['is_witness'] == 1][['speaker_first', 'speaker_last']].drop_duplicates().head(5)
        print(witness_examples.to_string(index=False))
    
    return congressional_data


def create_narrative_sankey(topic='', min_cluster_size=50, top_n=15):
    """
    Create a publication-ready Sankey diagram showing narrative flow from Reddit and Congressional sources to topics.
    Optimized for academic journals with sophisticated styling and maximum information density.
    
    Parameters:
        topic (str): Topic suffix for file naming (e.g., '_abortion', '_gmo')
        min_cluster_size (int): Minimum cluster size to include
        top_n (int): Maximum number of topics to show
    """
    # Construct file paths
    topic_prefix = f'_{topic}' if topic else ''
    csv_path = f"/home/arman/nature/clusters_to_visualize{topic_prefix}.csv"
    output_path = f"/home/arman/nature/narrative_flow_sankey{topic_prefix}.png"
    
    if not os.path.exists(csv_path):
        print(f"Error: Cluster file not found: {csv_path}")
        return
    
    # Read and filter data
    df = pd.read_csv(csv_path, encoding='latin1', on_bad_lines='warn')
    df_filtered = df[(df['cluster_size'] >= min_cluster_size) & (df['cluster'] != -1)]
    
    # Only show topics where o3_short_label is not NaN
    if 'o3_short_label' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['o3_short_label'].notna()]
        print(f"Filtered to {len(df_filtered)} topics with valid o3_short_label")
    
    df_filtered = df_filtered.nlargest(top_n, 'cluster_size')
    
    if len(df_filtered) == 0:
        print(f"No clusters meet criteria (min_size={min_cluster_size})")
        return
    
    # Academic color palette - sophisticated and print-friendly
    academic_colors = {
        'reddit': '#C75B27',      # Burnt orange - warm, professional
        'congress': '#1B3B6F',    # Navy blue - authoritative, institutional
        'high_overlap': '#2E7E32',  # Forest green - balanced, harmonious
        'med_overlap': '#F57C00',   # Amber - transitional, emerging
        'low_overlap': '#616161',   # Dark gray - single-platform
        'link_reddit': 'rgba(199, 91, 39, 0.6)',     # Semi-transparent burnt orange
        'link_congress': 'rgba(27, 59, 111, 0.6)',   # Semi-transparent navy
    }
    
    # Calculate statistics for enhanced information display
    total_reddit = int(df_filtered['total_reddit_arguments'].iloc[0])
    total_congress = int(df_filtered['total_congress_arguments'].iloc[0])
    total_clusters = len(df_filtered)
    
    # Calculate overlap distribution for dynamic thresholds
    overlap_scores = df_filtered['overlap_score'].values
    overlap_q75 = np.percentile(overlap_scores, 75)
    overlap_q50 = np.percentile(overlap_scores, 50)
    mean_overlap = np.mean(overlap_scores)
    
    # Create clean, readable node labels with essential information only
    node_labels = [
        f"Reddit<br>{total_reddit:,}",
        f"Congressional<br>{total_congress:,}"
    ]
    node_colors = [academic_colors['reddit'], academic_colors['congress']]
    
    # Enhanced topic node creation with minimal, essential information
    sorted_df = df_filtered.sort_values('overlap_score', ascending=False)
    
    for idx, row in sorted_df.iterrows():
        # Use proper topic label (fix the o3_short_label issue)
        if 'o3_short_label' in row and pd.notna(row['o3_short_label']):
            base_label = str(row['o3_short_label']).strip()
        else:
            base_label = f"Topic {row['cluster']}"
        
        # Keep labels concise - just the topic name and size
        cluster_size = int(row['cluster_size'])
        enhanced_label = f"{base_label}<br>n={cluster_size:,}"
        
        node_labels.append(enhanced_label)
        
        # Sophisticated color coding based on dynamic thresholds
        overlap_pct = row['overlap_score']
        if overlap_pct >= overlap_q75:
            node_colors.append(academic_colors['high_overlap'])
        elif overlap_pct >= overlap_q50:
            node_colors.append(academic_colors['med_overlap'])
        else:
            node_colors.append(academic_colors['low_overlap'])
    
    # Create enhanced links with variable opacity based on strength
    sources, targets, values, link_colors = [], [], [], []
    
    for idx, (_, row) in enumerate(sorted_df.iterrows()):
        topic_idx = idx + 2
        
        # Reddit to topic flows
        if row['reddit_arguments_in_cluster'] > 0:
            sources.append(0)
            targets.append(topic_idx)
            values.append(row['reddit_pct'])
            # Variable opacity based on flow strength
            opacity = 0.4 + (row['reddit_pct'] / 100) * 0.4  # 0.4-0.8 range
            link_colors.append(f"rgba(199, 91, 39, {opacity:.2f})")
        
        # Congressional to topic flows
        if row['congress_arguments_in_cluster'] > 0:
            sources.append(1)
            targets.append(topic_idx)
            values.append(row['congress_pct'])
            # Variable opacity based on flow strength
            opacity = 0.4 + (row['congress_pct'] / 100) * 0.4  # 0.4-0.8 range
            link_colors.append(f"rgba(27, 59, 111, {opacity:.2f})")
    
    # Create publication-quality Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='perpendicular',  # Better for academic layout
        node=dict(
            pad=30,                   # More padding to prevent overlap
            thickness=35,             # Thicker nodes for better visibility
            line=dict(color="#2C2C2C", width=1.5),  # Slightly thicker borders
            label=node_labels,
            color=node_colors,
            hovertemplate='<b>%{label}</b><extra></extra>'  # Simplified hover
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>' +
                         'Flow: %{value:.1f}%<extra></extra>'  # Simplified hover
        ),
        textfont=dict(size=16, family='Arial, sans-serif', color='black')  # Large, readable font for all text
    )])
    
    # Calculate aggregate statistics for subtitle
    total_represented_reddit = sum(row['reddit_arguments_in_cluster'] for _, row in df_filtered.iterrows())
    total_represented_congress = sum(row['congress_arguments_in_cluster'] for _, row in df_filtered.iterrows())
    coverage_reddit = (total_represented_reddit / total_reddit) * 100
    coverage_congress = (total_represented_congress / total_congress) * 100
    
    # Enhanced academic layout
    topic_display = topic.replace('_', ' ').title() if topic else 'Cross-Platform Discourse'
    
    fig.update_layout(
        title={
            'text': f"<b>Cross-Platform Discourse: {topic_display}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        font={'size': 14, 'family': 'Arial, sans-serif'},  # Larger base font
        height=700,   # Taller for better spacing
        width=1200,   # Wider for better node separation
        margin=dict(l=100, r=100, t=100, b=60),  # More margin for spacing
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Remove all annotations to reduce clutter - let the diagram speak for itself
    )
    
    # Save with high resolution for publication
    fig.write_image(output_path, width=1800, height=1050, scale=3, engine='kaleido')
    print(f"Clean, high-visibility Sankey diagram saved to: {output_path}")
    print(f"Resolution: 1800×1050 pixels at 3x scale (publication quality)")
    print(f"Topics displayed: {total_clusters} (minimum size: {min_cluster_size})")
    print(f"Streamlined design with large fonts and clear spacing for maximum readability")


def main():
    """
    Main function to run the analysis for a specified topic.
    """
    parser = argparse.ArgumentParser(description='Calculate representation analysis for different topics')
    parser.add_argument('--topic', type=str, default='abortion', 
                       choices=['abortion', 'gmo', 'nuclear', 'gun_control'],
                       help='Topic to analyze')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='Sample size for UMAP visualization (per dataset)')
    parser.add_argument('--similarity_threshold', type=float, default=0.70,
                       help='Similarity threshold for representation calculation')
    parser.add_argument('--create_sankey', action='store_true',
                       help='Create Sankey diagram visualization')
    parser.add_argument('--sankey_min_size', type=int, default=40,
                       help='Minimum cluster size for Sankey diagram')
    parser.add_argument('--sankey_top_n', type=int, default=15,
                       help='Maximum number of topics to show in Sankey')
    parser.add_argument('--consolidate_topics', action='store_true',
                       help='Enable topic consolidation before clustering')
    parser.add_argument('--consolidation_method', type=str, default='hybrid',
                       choices=['name_similarity', 'content_similarity', 'bertopic', 'hybrid'],
                       help='Method for topic consolidation')
    parser.add_argument('--consolidation_threshold', type=float, default=0.7,
                       help='Similarity threshold for topic consolidation')
    
    args = parser.parse_args()
    
    # Create Sankey diagram only if requested
    if args.create_sankey:
        print("\nCreating Sankey diagram...")
        create_narrative_sankey(
            topic=args.topic,
            min_cluster_size=args.sankey_min_size,
            top_n=args.sankey_top_n
        )
        print("\nSankey diagram creation complete!")
        return
    
    # Load and preprocess data
    congressional_data, reddit_data = load_and_preprocess_data(args.topic)
    
    # Set global variables for consolidation settings
    globals()['consolidation_enabled'] = args.consolidate_topics
    globals()['consolidation_method'] = args.consolidation_method
    globals()['consolidation_threshold'] = args.consolidation_threshold
    
    # Generate UMAP visualization
    print("\nGenerating UMAP visualization...")
    generate_umap_visualization_balanced(
        congressional_data, 
        reddit_data, 
        sample_size=args.sample_size,
        num_components=2,
        congress_length=len(congressional_data),
        reddit_length=len(reddit_data),
        topic=args.topic
    )
    
    # Calculate representation
    print("\nCalculating member representation...")
    calculate_representation(
        congressional_data, 
        reddit_data, 
        entity_type='member', 
        similarity_threshold=args.similarity_threshold,
        topic=args.topic
    )
    
    print("\nCalculating witness representation...")
    calculate_representation(
        congressional_data, 
        reddit_data, 
        entity_type='witness', 
        similarity_threshold=args.similarity_threshold,
        topic=args.topic
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()