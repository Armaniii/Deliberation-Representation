import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

class AutomaticTopicConsolidator:
    """
    Automatically consolidate fragmented topics into higher-level categories
    using multiple signals: topic names, document content, and embeddings.
    """
    
    def __init__(self, embedding_model=None, similarity_threshold=0.7):
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.similarity_threshold = similarity_threshold
        self.topic_embeddings = None
        self.consolidation_map = {}
        
    def consolidate_topics(self, data, method='hybrid'):
        """
        Main method to consolidate topics.
        
        Args:
            data: DataFrame with 'extracted_topic' and 'text_segment' columns
            method: 'name_similarity', 'content_similarity', 'bertopic', or 'hybrid'
        
        Returns:
            data: DataFrame with new 'consolidated_topic' column
            consolidation_info: Dictionary with consolidation details
        """
        
        if method == 'name_similarity':
            return self._consolidate_by_name_similarity(data)
        elif method == 'content_similarity':
            return self._consolidate_by_content_similarity(data)
        elif method == 'bertopic':
            return self._consolidate_by_bertopic(data)
        elif method == 'hybrid':
            return self._consolidate_hybrid(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _consolidate_by_name_similarity(self, data):
        """
        Method 1: Cluster topics based on semantic similarity of topic names.
        """
        print("=== CONSOLIDATING BY TOPIC NAME SIMILARITY ===")
        
        # Get unique topics
        unique_topics = data['extracted_topic'].unique()
        print(f"Found {len(unique_topics)} unique topics")
        
        # Embed topic names
        print("Embedding topic names...")
        topic_embeddings = self.embedding_model.encode(unique_topics)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(topic_embeddings)
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-self.similarity_threshold,
            metric='cosine',
            linkage='average'
        )
        
        clusters = clustering.fit_predict(topic_embeddings)
        
        # Create consolidation map
        consolidation_map = {}
        consolidated_topics = []
        
        for cluster_id in np.unique(clusters):
            cluster_topics = unique_topics[clusters == cluster_id]
            
            # Choose representative topic (longest common substring or most frequent)
            if len(cluster_topics) == 1:
                representative = cluster_topics[0]
            else:
                # Use the most frequent topic
                topic_counts = data[data['extracted_topic'].isin(cluster_topics)]['extracted_topic'].value_counts()
                representative = topic_counts.index[0]
            
            consolidated_topics.append({
                'consolidated_topic': representative,
                'original_topics': list(cluster_topics),
                'num_topics': len(cluster_topics)
            })
            
            for topic in cluster_topics:
                consolidation_map[topic] = representative
        
        # Apply consolidation
        data['consolidated_topic'] = data['extracted_topic'].map(consolidation_map)
        
        print(f"\nConsolidated {len(unique_topics)} topics into {len(np.unique(clusters))} groups")
        
        return data, {'method': 'name_similarity', 'groups': consolidated_topics}
    
    def _consolidate_by_content_similarity(self, data):
        """
        Method 2: Cluster topics based on the similarity of documents within them.
        """
        print("=== CONSOLIDATING BY CONTENT SIMILARITY ===")
        
        unique_topics = data['extracted_topic'].unique()
        
        # Speed optimization: Use pre-computed embeddings if available
        topic_representations = {}
        
        if 'embeddings' in data.columns:
            print("Using pre-computed embeddings for speed optimization...")
            # Vectorized approach using existing embeddings
            for topic in tqdm(unique_topics, desc="Creating topic representations"):
                topic_mask = data['extracted_topic'] == topic
                topic_embeddings = np.stack(data[topic_mask]['embeddings'].values)
                topic_representations[topic] = np.mean(topic_embeddings, axis=0)
        else:
            print("Computing embeddings from text...")
            # Batch process all unique documents first
            all_docs = []
            doc_to_topic = {}
            
            for topic in unique_topics:
                topic_docs = data[data['extracted_topic'] == topic]['text_segment'].tolist()
                # Sample max 20 docs per topic for speed
                if len(topic_docs) > 20:
                    topic_docs = np.random.choice(topic_docs, 20, replace=False).tolist()
                
                for doc in topic_docs:
                    all_docs.append(doc)
                    doc_to_topic[doc] = topic
            
            # Batch encode all documents at once
            print(f"Batch encoding {len(all_docs)} documents...")
            all_embeddings = self.embedding_model.encode(all_docs, show_progress_bar=True)
            
            # Group by topic and average
            for i, doc in enumerate(all_docs):
                topic = doc_to_topic[doc]
                if topic not in topic_representations:
                    topic_representations[topic] = []
                topic_representations[topic].append(all_embeddings[i])
            
            # Average embeddings per topic
            for topic in topic_representations:
                topic_representations[topic] = np.mean(topic_representations[topic], axis=0)
        
        # Convert to arrays for clustering
        topics_list = list(topic_representations.keys())
        embeddings_array = np.array([topic_representations[t] for t in topics_list])
        
        print("Performing hierarchical clustering...")
        # Cluster topic representations
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-self.similarity_threshold,
            metric='cosine',
            linkage='average'
        )
        
        clusters = clustering.fit_predict(embeddings_array)
        
        # Create consolidation map
        consolidation_info = self._create_consolidation_map(topics_list, clusters, data)
        
        # Apply consolidation
        data['consolidated_topic'] = data['extracted_topic'].map(consolidation_info['map'])
        
        return data, consolidation_info
    
    def _consolidate_by_bertopic(self, data):
        """
        Method 3: Use BERTopic to discover meta-topics from topic names + sample docs.
        """
        print("=== CONSOLIDATING USING BERTOPIC META-TOPICS ===")
        
        # Create synthetic documents for each topic
        synthetic_docs = []
        topic_labels = []
        
        for topic in data['extracted_topic'].unique():
            topic_data = data[data['extracted_topic'] == topic]
            
            # Combine topic name with sample documents
            sample_docs = topic_data['text_segment'].sample(min(5, len(topic_data))).tolist()
            synthetic_doc = f"{topic}: {' '.join(sample_docs[:3])}"
            
            synthetic_docs.append(synthetic_doc)
            topic_labels.append(topic)
        
        # Run BERTopic on synthetic documents
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=2,
            nr_topics='auto',
            calculate_probabilities=False
        )
        
        meta_topics, _ = topic_model.fit_transform(synthetic_docs)
        
        # Create consolidation map
        consolidation_map = {}
        consolidated_groups = defaultdict(list)
        
        for topic, meta_topic in zip(topic_labels, meta_topics):
            if meta_topic != -1:
                # Get keywords for this meta-topic
                keywords = topic_model.get_topic(meta_topic)
                meta_name = '_'.join([w[0] for w in keywords[:3]])
                consolidation_map[topic] = f"meta_{meta_name}"
                consolidated_groups[f"meta_{meta_name}"].append(topic)
            else:
                consolidation_map[topic] = topic  # Keep original if outlier
                consolidated_groups[topic].append(topic)
        
        # Apply consolidation
        data['consolidated_topic'] = data['extracted_topic'].map(consolidation_map)
        
        print(f"\nFound {len(set(meta_topics)) - 1} meta-topics (plus outliers)")
        
        return data, {
            'method': 'bertopic',
            'groups': dict(consolidated_groups),
            'model': topic_model
        }
    
    def _consolidate_hybrid(self, data):
        """
        Method 4: Hybrid approach combining multiple signals.
        Now includes stance-awareness and narrative flow detection.
        """
        print("=== HYBRID TOPIC CONSOLIDATION (STANCE-AWARE) ===")
        
        unique_topics = data['extracted_topic'].unique()
        n_topics = len(unique_topics)
        
        # 1. Name similarity
        print("Step 1: Computing name similarity...")
        name_embeddings = self.embedding_model.encode(unique_topics)
        name_similarity = cosine_similarity(name_embeddings)
        
        # 2. Content similarity - SPEED OPTIMIZED
        print("Step 2: Computing content similarity...")
        topic_representations = {}
        
        if 'embeddings' in data.columns:
            print("Using pre-computed embeddings...")
            # Use existing embeddings for massive speed improvement
            for topic in tqdm(unique_topics, desc="Computing topic centroids"):
                topic_mask = data['extracted_topic'] == topic
                if topic_mask.sum() > 0:
                    topic_embeddings = np.stack(data[topic_mask]['embeddings'].values)
                    topic_representations[topic] = np.mean(topic_embeddings, axis=0)
        else:
            print("Computing embeddings from text...")
            for topic in tqdm(unique_topics, desc="Computing topic representations"):
                topic_docs = data[data['extracted_topic'] == topic]['text_segment'].tolist()
                if len(topic_docs) > 0:
                    # Sample documents if too many
                    if len(topic_docs) > 20:  # Reduced from 50 for speed
                        topic_docs = np.random.choice(topic_docs, 20, replace=False)
                    
                    doc_embeddings = self.embedding_model.encode(topic_docs, show_progress_bar=False)
                    topic_representations[topic] = np.mean(doc_embeddings, axis=0)
        
        # Create content similarity matrix - VECTORIZED
        print("Computing content similarity matrix...")
        content_similarity = np.zeros((n_topics, n_topics))
        
        # Vectorized similarity computation for speed
        valid_topics = [t for t in unique_topics if t in topic_representations]
        if len(valid_topics) > 0:
            valid_embeddings = np.array([topic_representations[t] for t in valid_topics])
            similarity_matrix = cosine_similarity(valid_embeddings)
            
            # Map back to full matrix
            topic_to_idx = {t: i for i, t in enumerate(unique_topics)}
            for i, topic_i in enumerate(valid_topics):
                for j, topic_j in enumerate(valid_topics):
                    orig_i = topic_to_idx[topic_i]
                    orig_j = topic_to_idx[topic_j]
                    content_similarity[orig_i, orig_j] = similarity_matrix[i, j]
        
        # 3. Structural similarity (substring relationships)
        print("Step 3: Computing structural similarity...")
        structural_similarity = np.zeros((n_topics, n_topics))
        
        for i, topic_i in enumerate(unique_topics):
            for j, topic_j in enumerate(unique_topics):
                # Check if one topic is substring of another
                if topic_i in topic_j or topic_j in topic_i:
                    structural_similarity[i, j] = 1.0
                # Check for common words
                else:
                    words_i = set(topic_i.lower().split())
                    words_j = set(topic_j.lower().split())
                    if len(words_i) > 0 and len(words_j) > 0:
                        jaccard = len(words_i & words_j) / len(words_i | words_j)
                        structural_similarity[i, j] = jaccard
        
        # 4. Stance compatibility (NEW)
        print("Step 4: Computing stance compatibility...")
        stance_compatibility = np.ones((n_topics, n_topics))  # Default: compatible
        
        # Analyze stance distribution for each topic
        topic_stance_distributions = {}
        for topic in tqdm(unique_topics, desc="Analyzing stance distributions"):
            topic_data = data[data['extracted_topic'] == topic]
            stance_dist = topic_data['stance'].value_counts(normalize=True).to_dict()
            topic_stance_distributions[topic] = stance_dist
            
        # Check stance compatibility
        for i, topic_i in enumerate(unique_topics):
            for j, topic_j in enumerate(unique_topics):
                if i != j:
                    dist_i = topic_stance_distributions[topic_i]
                    dist_j = topic_stance_distributions[topic_j]
                    
                    # Check if topics have opposing dominant stances
                    dominant_i = max(dist_i, key=dist_i.get) if dist_i else None
                    dominant_j = max(dist_j, key=dist_j.get) if dist_j else None
                    
                    # Define opposing stance pairs
                    opposing_pairs = [
                        ('Argument_For', 'Argument_Against'),
                        ('pro', 'con'),
                        ('support', 'oppose')
                    ]
                    
                    # Check if dominant stances are opposing
                    for pair in opposing_pairs:
                        if (dominant_i in pair and dominant_j in pair and dominant_i != dominant_j):
                            stance_compatibility[i, j] = 0.0  # Incompatible
                            break
                    
                    # Also check for mixed vs pure topics
                    entropy_i = -sum(p * np.log(p + 1e-10) for p in dist_i.values())
                    entropy_j = -sum(p * np.log(p + 1e-10) for p in dist_j.values())
                    
                    # If one topic is pure (low entropy) and other is mixed, reduce compatibility
                    if abs(entropy_i - entropy_j) > 0.5:
                        stance_compatibility[i, j] *= 0.5
        
        # 5. Narrative flow detection (NEW)
        print("Step 5: Detecting narrative relationships...")
        narrative_relationships = np.zeros((n_topics, n_topics))
        
        # Define narrative flow patterns
        narrative_chains = [
            ['abortion', 'reproductive', 'women', 'health'],
            ['gun', 'control', 'safety', 'violence'],
            ['climate', 'change', 'environment', 'policy'],
            ['economy', 'jobs', 'employment', 'growth']
        ]
        
        # Detect if topics are part of same narrative chain
        for i, topic_i in enumerate(unique_topics):
            for j, topic_j in enumerate(unique_topics):
                topic_i_lower = topic_i.lower()
                topic_j_lower = topic_j.lower()
                
                # Check if topics share narrative chain
                for chain in narrative_chains:
                    chain_matches_i = sum(1 for word in chain if word in topic_i_lower)
                    chain_matches_j = sum(1 for word in chain if word in topic_j_lower)
                    
                    if chain_matches_i > 0 and chain_matches_j > 0:
                        # Topics are in same narrative chain
                        narrative_relationships[i, j] = 0.5 + 0.5 * min(chain_matches_i, chain_matches_j) / 2
                        
                # Check for causal/sequential keywords
                causal_keywords = ['leads to', 'causes', 'results in', 'because of', 'due to']
                if any(keyword in topic_i_lower or keyword in topic_j_lower for keyword in causal_keywords):
                    narrative_relationships[i, j] += 0.3
        
        # 6. Combine similarities (UPDATED with stance and narrative)
        print("Step 6: Combining all signals...")
        combined_similarity = (
            0.2 * name_similarity + 
            0.3 * content_similarity + 
            0.1 * structural_similarity +
            0.25 * stance_compatibility +  # Strong weight on stance
            0.15 * narrative_relationships  # Boost for narrative chains
        )
        
        # 7. Graph-based clustering
        print("Step 7: Graph-based consolidation with stance awareness...")
        
        # Create graph where edges exist if similarity > threshold
        G = nx.Graph()
        G.add_nodes_from(range(n_topics))
        
        for i in range(n_topics):
            for j in range(i+1, n_topics):
                if combined_similarity[i, j] > self.similarity_threshold:
                    G.add_edge(i, j, weight=combined_similarity[i, j])
        
        # Find connected components (natural groups)
        components = list(nx.connected_components(G))
        
        # Create consolidation map
        consolidation_map = {}
        consolidated_groups = []
        
        for component in components:
            component_topics = [unique_topics[i] for i in component]
            
            if len(component_topics) == 1:
                representative = component_topics[0]
            else:
                # Choose representative based on multiple criteria
                candidates = []
                
                for topic in component_topics:
                    score = 0
                    # Prefer shorter names (more general)
                    score -= len(topic) * 0.1
                    # Prefer topics with more documents
                    score += np.log(len(data[data['extracted_topic'] == topic]) + 1)
                    # Prefer topics without special characters
                    score -= topic.count(' - ') + topic.count(' and ') + topic.count(' in ')
                    
                    candidates.append((topic, score))
                
                representative = max(candidates, key=lambda x: x[1])[0]
            
            consolidated_groups.append({
                'representative': representative,
                'members': component_topics,
                'size': len(component_topics)
            })
            
            for topic in component_topics:
                consolidation_map[topic] = representative
        
        # Apply consolidation
        data['consolidated_topic'] = data['extracted_topic'].map(consolidation_map)
        
        print(f"\nConsolidated {n_topics} topics into {len(components)} groups")
        
        # Analyze consolidation results
        stance_coherent_groups = 0
        narrative_chain_groups = 0
        
        for group in consolidated_groups:
            members = group['members']
            if len(members) > 1:
                # Check stance coherence
                stances = []
                for topic in members:
                    topic_stances = data[data['extracted_topic'] == topic]['stance'].value_counts()
                    if len(topic_stances) > 0:
                        dominant_stance = topic_stances.idxmax()
                        stances.append(dominant_stance)
                
                # Check if all members have compatible stances
                opposing_found = False
                for i in range(len(stances)):
                    for j in range(i+1, len(stances)):
                        if ((stances[i] == 'Argument_For' and stances[j] == 'Argument_Against') or
                            (stances[i] == 'Argument_Against' and stances[j] == 'Argument_For')):
                            opposing_found = True
                            break
                
                if not opposing_found:
                    stance_coherent_groups += 1
                
                # Check narrative chain membership
                for chain in narrative_chains:
                    chain_members = sum(1 for topic in members 
                                      if any(word in topic.lower() for word in chain))
                    if chain_members > 1:
                        narrative_chain_groups += 1
                        break
        
        # Show some examples
        print("\nExample consolidations:")
        for group in sorted(consolidated_groups, key=lambda x: x['size'], reverse=True)[:5]:
            if group['size'] > 1:
                print(f"\n'{group['representative']}' includes:")
                for member in group['members'][:5]:
                    if member != group['representative']:
                        # Add stance info
                        topic_stances = data[data['extracted_topic'] == member]['stance'].value_counts()
                        stance_info = ""
                        if len(topic_stances) > 0:
                            dominant = topic_stances.idxmax()
                            pct = topic_stances.iloc[0] / topic_stances.sum() * 100
                            stance_info = f" [{dominant}: {pct:.0f}%]"
                        print(f"  - {member}{stance_info}")
        
        print(f"\n✅ Stance-coherent groups: {stance_coherent_groups}/{len(consolidated_groups)}")
        print(f"📊 Narrative chain groups: {narrative_chain_groups}/{len(consolidated_groups)}")
        
        # Add narrative flow information with directional chains
        data['narrative_chain'] = 'standalone'
        
        # Define directional narrative chains with progression patterns
        directional_chains = [
            {
                'keywords': ['abortion', 'reproductive', 'women', 'health'],
                'progression': [
                    ('health', ['health', 'medical', 'healthcare']),
                    ('womens_rights', ['women', 'womens', 'female', 'gender']),
                    ('reproductive_rights', ['reproductive', 'reproduction', 'fertility']),
                    ('abortion_access', ['abortion', 'terminate', 'choice'])
                ],
                'causal_direction': 'health → womens_rights → reproductive_rights → abortion_access'
            },
            {
                'keywords': ['gun', 'control', 'safety', 'violence'],
                'progression': [
                    ('violence_concern', ['violence', 'crime', 'shooting', 'deaths']),
                    ('public_safety', ['safety', 'protect', 'security']),
                    ('gun_regulation', ['control', 'regulation', 'laws']),
                    ('policy_implementation', ['policy', 'implementation', 'enforcement'])
                ],
                'causal_direction': 'violence_concern → public_safety → gun_regulation → policy_implementation'
            },
            {
                'keywords': ['climate', 'change', 'environment', 'policy'],
                'progression': [
                    ('environmental_concern', ['climate', 'environment', 'warming']),
                    ('scientific_evidence', ['science', 'research', 'evidence']),
                    ('policy_development', ['policy', 'regulation', 'action']),
                    ('implementation', ['implementation', 'practice', 'solutions'])
                ],
                'causal_direction': 'environmental_concern → scientific_evidence → policy_development → implementation'
            },
            {
                'keywords': ['economy', 'jobs', 'employment', 'growth'],
                'progression': [
                    ('economic_conditions', ['economy', 'economic', 'market']),
                    ('employment_impact', ['jobs', 'employment', 'workers']),
                    ('policy_response', ['policy', 'government', 'intervention']),
                    ('growth_outcomes', ['growth', 'development', 'prosperity'])
                ],
                'causal_direction': 'economic_conditions → employment_impact → policy_response → growth_outcomes'
            }
        ]
        
        # Analyze each topic and assign to narrative progression
        topic_to_chain = {}
        
        for topic in unique_topics:
            topic_lower = topic.lower()
            best_match = None
            best_score = 0
            best_stage = None
            
            # Check each directional chain
            for chain in directional_chains:
                # First check if topic matches any keywords in this chain
                keyword_matches = sum(1 for keyword in chain['keywords'] if keyword in topic_lower)
                
                if keyword_matches > 0:
                    # Find which stage of progression this topic belongs to
                    stage_scores = []
                    
                    for stage_name, stage_keywords in chain['progression']:
                        stage_score = sum(1 for keyword in stage_keywords if keyword in topic_lower)
                        if stage_score > 0:
                            stage_scores.append((stage_name, stage_score))
                    
                    if stage_scores:
                        # Get the stage with highest score
                        best_stage_for_chain = max(stage_scores, key=lambda x: x[1])
                        total_score = keyword_matches + best_stage_for_chain[1]
                        
                        if total_score > best_score:
                            best_match = chain
                            best_score = total_score
                            best_stage = best_stage_for_chain[0]
            
            # Assign topic to narrative chain with directional information
            if best_match and best_stage:
                # Create directional chain name showing progression
                chain_direction = best_match['causal_direction']
                # Highlight the current stage in the progression
                highlighted_direction = chain_direction.replace(best_stage, f"[{best_stage}]")
                topic_to_chain[topic] = highlighted_direction
            else:
                topic_to_chain[topic] = 'standalone'
        
        # Apply the narrative chain assignments
        for topic in unique_topics:
            chain_assignment = topic_to_chain.get(topic, 'standalone')
            data.loc[data['extracted_topic'] == topic, 'narrative_chain'] = chain_assignment
        
        # Print some examples of directional assignments
        print(f"\n📊 Directional Narrative Chain Examples:")
        directional_examples = [(topic, chain) for topic, chain in topic_to_chain.items() 
                              if chain != 'standalone'][:5]
        for topic, chain in directional_examples:
            print(f"  • {topic} → {chain}")
        
        standalone_count = sum(1 for chain in topic_to_chain.values() if chain == 'standalone')
        directional_count = len(topic_to_chain) - standalone_count
        print(f"\n✅ Narrative Analysis: {directional_count} topics in directional chains, {standalone_count} standalone")
        
        return data, {
            'method': 'hybrid',
            'groups': consolidated_groups,
            'similarity_matrix': combined_similarity,
            'consolidation_map': consolidation_map,
            'stance_compatibility_matrix': stance_compatibility,
            'narrative_relationships_matrix': narrative_relationships,
            'stance_coherent_groups': stance_coherent_groups,
            'narrative_chain_groups': narrative_chain_groups
        }
    
    def _create_consolidation_map(self, topics_list, clusters, data):
        """Helper to create consolidation map from clustering results."""
        
        consolidation_map = {}
        groups = []
        
        for cluster_id in np.unique(clusters):
            cluster_topics = [topics_list[i] for i in range(len(topics_list)) if clusters[i] == cluster_id]
            
            # Choose representative
            topic_sizes = {t: len(data[data['extracted_topic'] == t]) for t in cluster_topics}
            representative = max(cluster_topics, key=lambda t: topic_sizes[t])
            
            groups.append({
                'consolidated_topic': representative,
                'original_topics': cluster_topics,
                'num_topics': len(cluster_topics),
                'total_docs': sum(topic_sizes.values())
            })
            
            for topic in cluster_topics:
                consolidation_map[topic] = representative
        
        return {
            'map': consolidation_map,
            'groups': groups,
            'num_original': len(topics_list),
            'num_consolidated': len(groups)
        }


def simple_topic_consolidation(data):
    """
    Simplest approach: Run BERTopic directly on topic names.
    """
    
    # Get unique topics with their document counts
    topic_info = data['extracted_topic'].value_counts().reset_index()
    topic_info.columns = ['extracted_topic', 'count']
    
    # Use topic names as "documents" for BERTopic
    topic_model = BERTopic(
        min_topic_size=2,
        nr_topics='auto'
    )
    
    # Weight by frequency (repeat topic names by their count)
    weighted_topics = []
    for _, row in topic_info.iterrows():
        # Add topic name multiple times based on document count
        weighted_topics.extend([row['extracted_topic']] * min(row['count'], 10))
    
    # Cluster topic names
    clusters, _ = topic_model.fit_transform(weighted_topics)
    
    # Create mapping
    unique_topics = topic_info['extracted_topic'].tolist()
    topic_clusters = clusters[:len(unique_topics)]
    
    consolidation_map = {}
    for topic, cluster in zip(unique_topics, topic_clusters):
        if cluster != -1:
            keywords = topic_model.get_topic(cluster)
            consolidated_name = '_'.join([w[0] for w in keywords[:3]])
            consolidation_map[topic] = consolidated_name
        else:
            consolidation_map[topic] = topic
    
    data['consolidated_topic'] = data['extracted_topic'].map(consolidation_map)
    
    return data