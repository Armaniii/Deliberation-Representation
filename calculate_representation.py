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
import argparse
import numpy as np
import pandas as pd
import cupy as cp
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import tqdm


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
        'congress_file': "/home/arman/apsa/congress_data/gun_control/hearings_gun_control_stance.csv",
        'reddit_file': "/home/arman/nature/reddit_data/gun_control/reddit_argument_analysis_results_all_gun_control.pkl",
        'output_prefix': 'gun_control'
    }
}


def parse_embedding(emb):
    """
    Parse string representation of embeddings to numpy array.
    
    Parameters:
        emb: Embedding that could be a string or array
        
    Returns:
        np.array or None: Parsed embedding array or None if parsing fails
    """
    try:
        # Handle empty or null values
        if emb is None or pd.isna(emb):
            return None
            
        # If already a numpy array, return as is
        if isinstance(emb, np.ndarray):
            return emb
            
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
    model = SentenceTransformer('/home/arman/vienna/models/contrastive_finetune_v2_mpnet-v2_o3')
    
    print(f"Starting embedding creation for {len(df)} rows")
    
    # Create mask for valid text segments
    valid_mask = df['text_segment'].notna() & (df['text_segment'].str.strip() != '')
    print(f"Found {valid_mask.sum()} valid text segments out of {len(df)} total")
    
    # Initialize embeddings column with None values (object dtype to store numpy arrays)
    df['embeddings'] = pd.Series([None] * len(df), dtype='object')
    
    if valid_mask.any():
        # Get valid text segments and filter out empty/short ones
        valid_texts = df.loc[valid_mask, 'text_segment'].tolist()
        
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
        
        # Get cluster centroid from UMAP dimensions
        centroid_umap = np.array([
            clusters_to_visualize.loc[clusters_to_visualize['cluster'] == cluster_id, 'UMAP1_mean'].values[0],
            clusters_to_visualize.loc[clusters_to_visualize['cluster'] == cluster_id, 'UMAP2_mean'].values[0]
        ])
        
        # Calculate distance to centroid for each point in UMAP space
        cluster_data = cluster_data.copy()  # Fix SettingWithCopyWarning
        cluster_data['distance_to_centroid'] = cluster_data.apply(
            lambda row: np.sqrt((row['UMAP1'] - centroid_umap[0])**2 + (row['UMAP2'] - centroid_umap[1])**2),
            axis=1
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

    # Sample data if necessary
    if len(congressional_data) > sample_size:
        congressional_data = congressional_data.sample(sample_size, random_state=42)
    if len(reddit_data) > sample_size:
        reddit_data = reddit_data.sample(sample_size, random_state=42)

    # Combine data
    all_data = pd.concat([congressional_data, reddit_data], ignore_index=True)
    
    # Ensure we have a text column
    if 'text_to_embed' not in all_data.columns:
        all_data['text_to_embed'] = all_data['text_segment']
    
    # Stack embeddings
    all_embeddings = np.stack(all_data['embeddings'].values)
    
    # Convert to CuPy arrays for GPU acceleration
    all_embeddings_gpu = cp.asarray(all_embeddings)
    
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cp.asnumpy(all_embeddings_gpu))
    
    # Perform UMAP
    print("Performing UMAP dimensionality reduction...")
    umap_model = UMAP(n_neighbors=15, n_components=num_components, min_dist=0.1, 
                      metric='cosine', random_state=42)
    umap_embeddings = umap_model.fit_transform(normalized_data)
    
    # Perform clustering
    print("Performing HDBSCAN clustering...")
    hdbscan_model = HDBSCAN(min_cluster_size=15, prediction_data=True)
    hdbscan_model.fit(umap_embeddings)
    
    # Add UMAP coordinates and cluster labels to dataframe
    all_data['UMAP1'] = umap_embeddings[:, 0]
    all_data['UMAP2'] = umap_embeddings[:, 1]
    all_data['cluster'] = hdbscan_model.labels_
    
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
            'Reddit': (x[x == 'Reddit'].count() / total_reddit) * 100
        },
        'extracted_topic': lambda x: x.value_counts().idxmax() if 'extracted_topic' in all_data.columns else 'N/A',
        'stance': lambda x: x.value_counts().idxmax() if 'stance' in all_data.columns else 'N/A',
        'UMAP1': ['mean', 'std'],
        'UMAP2': ['mean', 'std'],
    }).reset_index()
    
    # Flatten MultiIndex columns
    cluster_stats.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in cluster_stats.columns]
    
    # Rename columns
    cluster_stats.columns = [
        'cluster', 'source', 'extracted_topic', 'stance',
        'UMAP1_mean', 'UMAP1_std', 'UMAP2_mean', 'UMAP2_std'
    ]
    
    # Merge with cluster sizes and remove noise cluster
    cluster_stats = cluster_stats.merge(cluster_sizes, on='cluster')
    cluster_stats = cluster_stats[cluster_stats['cluster'] != -1]
    
    # Calculate subreddit representation if available
    if 'subreddit' in all_data.columns:
        cluster_stats['subreddit_representation'] = cluster_stats['cluster'].apply(
            lambda x: calculate_subreddit_representation(x, 'subreddit', all_data))
    
    # Calculate overlap score
    cluster_stats['overlap_score'] = cluster_stats['source'].apply(lambda x: calculate_overlap(x))
    
    # Get representative arguments
    representative_arguments = get_representative_arguments(all_data, cluster_stats, n=200)
    
    # Export cluster statistics and representative arguments
    topic_prefix = f'_{topic}' if topic else ''
    export_cluster_data(cluster_stats, representative_arguments, topic_prefix)
    
    print("UMAP visualization and clustering complete.")
    
    # Optionally create and save visualization figure
    # fig = create_visualization_figure(all_data, cluster_stats)
    # fig.write_image("/home/arman/nature/umap_visualization.png")


def export_cluster_data(cluster_stats, representative_arguments, topic_prefix=''):
    """
    Export cluster statistics and representative arguments to CSV.
    
    Parameters:
        cluster_stats (pd.DataFrame): Cluster statistics
        representative_arguments (pd.DataFrame): Representative arguments for each cluster
        topic_prefix (str): Topic prefix for output file naming (e.g., '_gmo', '_gun')
    """
    # Group representative arguments by cluster
    grouped_args = representative_arguments.groupby('cluster')['text_segment'].apply(
        lambda x: '||'.join(str(text).replace(',', '&#44;').replace('\n', ' ').replace('||', '&#124;&#124;') 
                           for text in x[:50])  # Limit to 50 arguments
    ).reset_index()
    
    # Create export dataframe
    export_data = pd.DataFrame()
    export_data['cluster'] = cluster_stats['cluster']
    export_data['source_distribution'] = cluster_stats['source']
    export_data['cluster_size'] = cluster_stats['cluster_size']
    export_data['overlap_score'] = cluster_stats['overlap_score']
    
    # Calculate percentages
    export_data['reddit_pct'] = cluster_stats['source'].apply(lambda x: x.get('Reddit', 0)).round(2)
    export_data['congress_pct'] = cluster_stats['source'].apply(lambda x: x.get('Congressional', 0)).round(2)
    export_data['overlap_percentage'] = export_data.apply(
        lambda row: min(row['reddit_pct'], row['congress_pct']), axis=1).round(2)
    
    # Merge with representative arguments
    export_data = export_data.merge(grouped_args, on='cluster', how='left')
    
    # Save to CSV with topic prefix
    output_file = f"/home/arman/nature/clusters_to_visualize{topic_prefix}.csv"
    export_data.to_csv(output_file, 
                      index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"Cluster data exported to {output_file}")


def calculate_representation(data, reddit_data, entity_type='member', similarity_threshold=0.70, topic=''):
    """
    Calculate per subreddit representation for each entity (member or witness).
    
    Parameters:
        data (pd.DataFrame): Congressional data with is_member and is_witness flags
        reddit_data (pd.DataFrame): Reddit data with subreddit information
        entity_type (str): Type of entity ('member' or 'witness')
        similarity_threshold (float): Cosine similarity threshold for matching
        
    Returns:
        pd.DataFrame: Statistics for each entity showing subreddit representation
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
    
    # Initialize results list and missing members tracking
    results = []
    missing_members = []
    
    # Process each entity
    for entity_key, entity_group in tqdm.tqdm(entity_filter.groupby(groupby_columns), 
                                           desc=f'Processing {entity_type}s'):
        try:
            # Stack embeddings
            if len(entity_group) == 1:
                entity_embeddings = np.array([entity_group['embeddings'].values[0]])
            else:
                entity_embeddings = np.stack(entity_group['embeddings'].values)
                
            total_args = entity_embeddings.shape[0]
            
            # Create initial stats dictionary
            if entity_type.lower() == 'member':
                entity_stats = {
                    'speaker_last': entity_key[0],
                    'speaker_first': entity_key[1],
                    'govtrack': entity_key[2],
                    'congress': entity_key[3],
                    'total_arguments': total_args
                }
            else:  # witness
                entity_stats = {
                    'speaker_last': entity_key[0],
                    'file_name': entity_key[1],
                    'total_arguments': total_args
                }
            
            overall_matched = np.zeros(total_args, dtype=bool)
            
            # Calculate similarity with each subreddit
            for subreddit in reddit_data['subreddit'].unique():
                subreddit_group = reddit_data[reddit_data['subreddit'] == subreddit]
                
                # Stack subreddit embeddings
                if len(subreddit_group) == 1:
                    subreddit_embeddings = np.array([subreddit_group['embeddings'].values[0]])
                else:
                    subreddit_embeddings = np.stack(subreddit_group['embeddings'].values)
                
                # Compute cosine similarity
                similarity_matrix = util.pytorch_cos_sim(entity_embeddings, subreddit_embeddings).numpy()
                
                # Find matches above threshold
                max_similarities = np.max(similarity_matrix, axis=1)
                matched = max_similarities > similarity_threshold
                overall_matched = np.logical_or(overall_matched, matched)
                
                # Calculate statistics
                matched_count = np.sum(matched)
                entity_stats[f'{subreddit}_matched'] = matched_count
                entity_stats[f'{subreddit}_percentage'] = (matched_count / total_args) * 100
            
            # Calculate overall match percentage
            overall_matched_count = np.sum(overall_matched)
            entity_stats['overall_match_percentage'] = (overall_matched_count / total_args) * 100
            
            # Track members with no matches
            if overall_matched_count == 0:
                missing_member_info = entity_stats.copy()
                missing_member_info['reason'] = 'No similarity matches found'
                missing_members.append(missing_member_info)
            
            results.append(entity_stats)
            
        except Exception as e:
            print(f"Error processing {entity_key}: {e}")
            continue
    
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
    
    # Create embeddings for congressional data
    congressional_data = create_embeddings(congressional_data)
    
    # Filter for arguments
    if "argument_prediction" in congressional_data.columns:
        congressional_data = congressional_data[congressional_data['argument_prediction'] == "Argument"]
    elif "label" in congressional_data.columns:
        congressional_data["argument_prediction"] = congressional_data["label"]
        congressional_data = congressional_data[congressional_data['argument_prediction'] == "Argument"]
    
    print(f"Congressional arguments: {len(congressional_data)}")
    
    # Load reddit data
    reddit_data = pd.read_pickle(config['reddit_file'])
    
    # Ensure text_segment column exists
    if "text_segment" not in reddit_data.columns:
        reddit_data["text_segment"] = reddit_data.get("processed_text", 
                                                     reddit_data.get("text", ""))
    
    # Filter for arguments
    if "argument_prediction" in reddit_data.columns:
        reddit_data = reddit_data[reddit_data['argument_prediction'] == "Argument"]
    elif "argument" in reddit_data.columns:
        reddit_data["argument_prediction"] = reddit_data["argument"]
        reddit_data = reddit_data[reddit_data['argument_prediction'] == "Argument"]
    
    # Create embeddings for reddit data
    reddit_data = create_embeddings(reddit_data)
    
    print(f"Reddit arguments: {len(reddit_data)}")
    
    # Process member and witness information for congressional data
    congressional_data = process_member_witness_data(congressional_data)
    
    return congressional_data, reddit_data


def process_member_witness_data(congressional_data):
    """
    Process and add member/witness flags to congressional data.
    
    Parameters:
        congressional_data (pd.DataFrame): Congressional hearing data
        
    Returns:
        pd.DataFrame: Congressional data with is_member and is_witness flags
    """
    # Create bioname
    if 'speaker_first' in congressional_data.columns and 'speaker_last' in congressional_data.columns:
        congressional_data['bioname'] = (congressional_data['speaker_first'].fillna('') + ' ' + 
                                        congressional_data['speaker_last'].fillna(''))
    
    # Load member data
    member_file = '/home/arman/apsa/congress_data/abortion/df_member_abortion.csv'
    if os.path.exists(member_file):
        df_members = pd.read_csv(member_file)
        df_members = df_members.drop_duplicates(subset=['congress', 'govtrack'])
        
        # Merge with members
        congressional_data = pd.merge(
            congressional_data, 
            df_members, 
            on=['congress', 'govtrack'], 
            how='left',
            suffixes=('', '_member')
        )
        congressional_data['is_member'] = np.where(congressional_data['bioname_member'].notna(), 1, 0)
    else:
        congressional_data['is_member'] = 0
    
    # Load witness data
    witness_file = '/home/arman/apsa/congress_data/abortion/df_witness_abortion.csv'
    if os.path.exists(witness_file):
        df_witness = pd.read_csv(witness_file)
        df_witness.rename(columns={'last_name': 'speaker_last'}, inplace=True)
        df_witness['speaker_last'] = df_witness['speaker_last'].str.capitalize()
        df_witness = df_witness.drop_duplicates(subset=['file_name', 'speaker_last'])
        
        # Merge with witnesses
        congressional_data = pd.merge(
            congressional_data, 
            df_witness, 
            on=['file_name', 'speaker_last'], 
            how='left',
            suffixes=('', '_witness')
        )
        congressional_data['is_witness'] = np.where(congressional_data['witness_info'].notna(), 1, 0)
    else:
        congressional_data['is_witness'] = 0
    
    # Assign member type
    congressional_data['Type'] = np.where(congressional_data.get('witness_info', pd.Series()).notna(), 
                                         'Witness', 'Member')
    
    print(f"Members identified: {congressional_data['is_member'].sum()}")
    print(f"Witnesses identified: {congressional_data['is_witness'].sum()}")
    
    return congressional_data


def main():
    """
    Main function to run the analysis for a specified topic.
    """
    parser = argparse.ArgumentParser(description='Calculate representation analysis for different topics')
    parser.add_argument('--topic', type=str, default='abortion', 
                       choices=['abortion', 'gmo', 'nuclear', 'gun_control'],
                       help='Topic to analyze')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='Sample size for UMAP visualization')
    parser.add_argument('--similarity_threshold', type=float, default=0.70,
                       help='Similarity threshold for representation calculation')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    congressional_data, reddit_data = load_and_preprocess_data(args.topic)
    
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
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()