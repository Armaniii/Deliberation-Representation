import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import networkx as nx
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import faiss
from scipy import sparse
import time
import gc

class AutomaticTopicConsolidator:
    """
    Automatically consolidate fragmented topics into higher-level categories
    using multiple signals: topic names, document content, and embeddings.
    
    DYNAMIC FEATURES (NEW):
    - BERTopic-based domain discovery replacing hardcoded patterns
    - Graph-based narrative chain discovery using embedding trajectories
    - Adaptive similarity computation with learned optimal weights
    - Finetuned embedding integration for semantic understanding
    
    HIERARCHICAL FEATURES:
    - Multi-level topic taxonomy preserving semantic granularity
    - Separate stance detection decoupled from topic consolidation
    - Generalizable domain filtering with auto-detection
    - Domain coherence analysis instead of stance-based grouping
    - Semantic representative selection for clusters
    
    Methods:
    - hierarchical_dynamic: Fully dynamic discovery with BERTopic (NEW)
    - dynamic_with_stance: Dynamic discovery + separate stance detection (NEW)
    - hierarchical: Multi-level topic taxonomy 
    - hybrid_with_stance: Hierarchical + separate stance detection
    - hybrid: Original method updated with domain coherence
    - name_similarity, content_similarity, bertopic: Original methods
    """
    
    def __init__(self, embedding_model=None, similarity_threshold=0.7):
        self.embedding_model = SentenceTransformer('/home/arman/vienna/models/contrastive_finetune_v2_mpnet-v2_mal')
        self.similarity_threshold = similarity_threshold
        self.topic_embeddings = None
        self.consolidation_map = {}
        
        # Performance optimization caches
        self._embedding_cache = {}
        self._similarity_cache = {}
        self._faiss_index = None
        self._cached_topics = None
        
    def _filter_domain_relevant_topics(self, topics, domain_keywords=None, 
                                      similarity_threshold=0.3, keep_short_topics=True):
        """
        Filter topics to keep only domain-relevant ones using flexible keyword matching
        and semantic similarity. Addresses critical issue of domain contamination.
        
        Args:
            topics: List of topic names to filter
            domain_keywords: List of keywords/phrases that define the target domain.
                           If None, performs automatic domain detection using topic frequency.
            similarity_threshold: Minimum semantic similarity to domain keywords (0-1)
            keep_short_topics: Whether to keep potentially general short topics
            
        Returns:
            List of domain-relevant topics
        """
        # If no keywords provided, auto-detect domain from most frequent topics
        if domain_keywords is None:
            print("No domain keywords provided. Auto-detecting domain from topic frequency...")
            domain_keywords = self._auto_detect_domain_keywords(topics)
            print(f"Auto-detected domain keywords: {domain_keywords[:10]}...")
        
        # Ensure domain_keywords is a list
        if isinstance(domain_keywords, str):
            domain_keywords = [domain_keywords]
            
        relevant_topics = []
        filtered_count = 0
        
        print(f"Filtering topics using {len(domain_keywords)} domain keywords/phrases")
        print(f"Semantic similarity threshold: {similarity_threshold}")
        
        # If we have many keywords, use embedding-based similarity for efficiency
        if len(domain_keywords) > 10:
            relevant_topics = self._filter_by_semantic_similarity(
                topics, domain_keywords, similarity_threshold, keep_short_topics
            )
            filtered_count = len(topics) - len(relevant_topics)
        else:
            # Use keyword matching for smaller keyword sets
            for topic in topics:
                topic_lower = topic.lower()
                
                # Check for keyword/phrase matches
                is_relevant = any(
                    keyword.lower() in topic_lower or topic_lower in keyword.lower()
                    for keyword in domain_keywords
                )
                
                if is_relevant:
                    relevant_topics.append(topic)
                elif keep_short_topics and len(topic_lower.split()) <= 2 and len(topic_lower) >= 3:
                    # Keep short topics that might be relevant abbreviations or general terms
                    relevant_topics.append(topic)
                else:
                    filtered_count += 1
        
        print(f"Filtered out {filtered_count} irrelevant topics")
        print(f"Retained {len(relevant_topics)} domain-relevant topics")
        
        return relevant_topics
    
    def _auto_detect_domain_keywords(self, topics, top_k=20):
        """
        Automatically detect domain keywords from the most frequent topic terms.
        
        Args:
            topics: List of topic names
            top_k: Number of top keywords to extract
            
        Returns:
            List of domain-representative keywords
        """
        # Extract all words from topics
        all_words = []
        for topic in topics:
            # Split and clean words
            words = [word.strip().lower() for word in topic.replace('_', ' ').split()]
            # Filter out very short words and common stop words
            words = [w for w in words if len(w) > 2 and w not in 
                    {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'own', 'say', 'she', 'too', 'use'}]
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords by frequency
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [word for word, count in top_keywords]
    
    def _filter_by_semantic_similarity(self, topics, domain_keywords, threshold, keep_short_topics):
        """
        Filter topics using semantic similarity to domain keywords.
        
        Args:
            topics: List of topic names
            domain_keywords: List of domain keywords
            threshold: Similarity threshold
            keep_short_topics: Whether to keep short topics
            
        Returns:
            List of relevant topics
        """
        print("Using semantic similarity filtering...")
        
        # Embed domain keywords and topics
        keyword_embeddings = self.embedding_model.encode(domain_keywords)
        topic_embeddings = self.embedding_model.encode(topics)
        
        # Calculate average domain embedding
        domain_centroid = np.mean(keyword_embeddings, axis=0)
        
        # Calculate similarities
        similarities = cosine_similarity(topic_embeddings, domain_centroid.reshape(1, -1)).flatten()
        
        relevant_topics = []
        for i, (topic, similarity) in enumerate(zip(topics, similarities)):
            if similarity >= threshold:
                relevant_topics.append(topic)
            elif keep_short_topics and len(topic.lower().split()) <= 2 and len(topic) >= 3:
                relevant_topics.append(topic)
        
        return relevant_topics
    
    def detect_stance_separately(self, data, stance_keywords=None, confidence_threshold=0.1):
        """
        Separate stance detection that doesn't interfere with topic consolidation.
        Returns stance classifications without forcing topics into pro/con categories.
        
        Args:
            data: DataFrame with 'extracted_topic' and 'text_segment' columns
            stance_keywords: Dict mapping stance labels to keyword lists. If None, uses defaults.
            confidence_threshold: Minimum confidence score to assign stance (vs 'neutral_unclear')
            
        Returns:
            data: DataFrame with new 'detected_stance' column
            stance_by_topic: DataFrame with stance distribution by topic
        """
        print("=== SEPARATE STANCE DETECTION ===")
        
        # Default stance detection patterns (can be customized per domain)
        if stance_keywords is None:
            stance_keywords = {
                'pro_position': [
                    'support', 'favor', 'agree', 'endorse', 'advocate', 'champion',
                    'benefits', 'advantages', 'positive', 'necessary', 'important',
                    'rights', 'freedom', 'choice', 'access', 'healthcare', 'autonomy'
                ],
                'anti_position': [
                    'oppose', 'against', 'reject', 'condemn', 'criticize', 'disagree',
                    'harmful', 'dangerous', 'wrong', 'immoral', 'unethical', 'illegal',
                    'ban', 'prohibit', 'restrict', 'prevent', 'stop', 'murder', 'killing'
                ],
                'neutral_medical': [
                    'procedure', 'medical', 'health', 'doctor', 'clinical', 'treatment',
                    'complication', 'diagnosis', 'therapy', 'surgery', 'medication',
                    'healthcare provider', 'patient', 'symptoms', 'recovery'
                ],
                'neutral_legal': [
                    'law', 'legal', 'court', 'constitutional', 'legislation', 'policy',
                    'regulation', 'judicial', 'statute', 'ruling', 'precedent',
                    'jurisdiction', 'litigation', 'compliance'
                ],
                'neutral_factual': [
                    'study', 'research', 'data', 'statistics', 'evidence', 'findings',
                    'analysis', 'report', 'survey', 'demographics', 'trends',
                    'methodology', 'results', 'conclusion'
                ]
            }
        
        # Classify each document's stance
        stance_classifications = []
        confidence_scores = []
        
        print(f"Classifying stance for {len(data)} documents...")
        print(f"Using confidence threshold: {confidence_threshold}")
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Detecting stance"):
            text = str(row['text_segment']).lower()
            topic = str(row['extracted_topic']).lower()
            combined_text = f"{text} {topic}"
            
            # Score each stance category
            scores = {}
            for stance, keywords in stance_keywords.items():
                # Count keyword matches and normalize by text length
                keyword_matches = sum(1 for keyword in keywords if keyword in combined_text)
                # Normalize by square root of text length to avoid bias toward longer texts
                normalized_score = keyword_matches / max(1, len(combined_text.split()) ** 0.5)
                scores[stance] = normalized_score
            
            # Determine stance based on highest score and confidence
            max_score = max(scores.values()) if scores.values() else 0
            
            if max_score < confidence_threshold:
                stance = 'neutral_unclear'
                confidence = 0.0
            else:
                stance = max(scores, key=scores.get)
                # Calculate confidence as ratio of max score to second-highest score
                sorted_scores = sorted(scores.values(), reverse=True)
                confidence = (sorted_scores[0] / (sorted_scores[1] + 1e-10)) if len(sorted_scores) > 1 else 1.0
            
            stance_classifications.append(stance)
            confidence_scores.append(confidence)
        
        # Add results to dataframe
        data['detected_stance'] = stance_classifications
        data['stance_confidence'] = confidence_scores
        
        # Generate stance statistics by topic
        print("\nGenerating stance statistics by topic...")
        stance_by_topic = data.groupby('extracted_topic')['detected_stance'].value_counts().unstack(fill_value=0)
        
        # Add percentage distributions
        stance_percentages = stance_by_topic.div(stance_by_topic.sum(axis=1), axis=0) * 100
        
        # Calculate stance diversity (entropy) for each topic
        stance_diversity = []
        for topic in stance_by_topic.index:
            topic_stances = stance_by_topic.loc[topic]
            proportions = topic_stances / topic_stances.sum()
            # Calculate Shannon entropy
            entropy = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
            stance_diversity.append(entropy)
        
        stance_by_topic['stance_diversity'] = stance_diversity
        
        # Show summary statistics
        print(f"\nStance Detection Summary:")
        print(f"Overall stance distribution:")
        overall_stance_dist = data['detected_stance'].value_counts()
        for stance, count in overall_stance_dist.items():
            pct = (count / len(data)) * 100
            print(f"  {stance}: {count} ({pct:.1f}%)")
        
        print(f"\nAverage stance confidence: {data['stance_confidence'].mean():.3f}")
        print(f"Topics with high stance diversity (>1.5): {sum(1 for d in stance_diversity if d > 1.5)}")
        
        return data, stance_by_topic
    
    def _discover_domains_dynamically(self, topics, n_domains=6, min_domain_size=3):
        """
        Dynamically discover semantic domains using BERTopic and finetuned embeddings.
        
        Args:
            topics: List of topic names
            n_domains: Target number of domains to discover
            min_domain_size: Minimum topics per domain
        
        Returns:
            discovered_domains: Dict mapping domain names to domain info
            domain_keywords: Auto-generated keywords for each domain
        """
        print("=== DYNAMIC DOMAIN DISCOVERY ===")
        
        if len(topics) < min_domain_size * 2:
            print(f"Too few topics ({len(topics)}) for domain discovery, using single domain")
            return {'general_domain': {'topics': list(topics), 'keywords': ['general'], 'centroid': None}}, {'general_domain': ['general']}
        
        # Step 1: Embed all topics using finetuned model
        print("Embedding topics with finetuned model...")
        topic_embeddings = self.embedding_model.encode(topics)
        
        # Step 2: Use BERTopic for domain clustering
        domain_model = BERTopic(
            embedding_model=self.embedding_model,  # Use finetuned model
            min_topic_size=min_domain_size,
            nr_topics=n_domains if n_domains > 0 else 'auto',
            calculate_probabilities=False,  # Faster computation
            verbose=False
        )
        
        try:
            # Fit BERTopic on topic names (treating them as "documents")
            domain_assignments, _ = domain_model.fit_transform(topics)
        except Exception as e:
            print(f"BERTopic failed: {e}, falling back to embedding-based clustering")
            return self._fallback_domain_discovery(topics, topic_embeddings, n_domains)
        
        # Step 3: Extract domain characteristics
        discovered_domains = {}
        domain_keywords = {}
        
        valid_domain_ids = [d for d in set(domain_assignments) if d != -1]
        if not valid_domain_ids:
            print("No valid domains found, using fallback clustering")
            return self._fallback_domain_discovery(topics, topic_embeddings, n_domains)
        
        for domain_id in valid_domain_ids:
            # Get topics in this domain
            domain_topic_indices = [i for i, d in enumerate(domain_assignments) if d == domain_id]
            domain_topics = [topics[i] for i in domain_topic_indices]
            
            if len(domain_topics) < min_domain_size:
                continue
                
            # Get BERTopic keywords for this domain
            try:
                topic_words = domain_model.get_topic(domain_id)
                keywords = [word for word, score in topic_words[:10]]
            except:
                keywords = self._extract_keywords_from_topics(domain_topics)
            
            # Generate semantic domain name using embedding similarity
            domain_name = self._generate_domain_name(domain_topics, keywords)
            
            # Calculate domain centroid
            domain_centroid = np.mean([topic_embeddings[i] for i in domain_topic_indices], axis=0)
            
            discovered_domains[domain_name] = {
                'topics': domain_topics,
                'keywords': keywords,
                'centroid': domain_centroid,
                'size': len(domain_topics)
            }
            domain_keywords[domain_name] = keywords
        
        # Handle outliers (topics assigned to -1)
        outlier_topics = [topics[i] for i, d in enumerate(domain_assignments) if d == -1]
        if outlier_topics:
            discovered_domains['miscellaneous'] = {
                'topics': outlier_topics,
                'keywords': ['miscellaneous', 'other'],
                'centroid': np.mean([topic_embeddings[i] for i, d in enumerate(domain_assignments) if d == -1], axis=0) if outlier_topics else None,
                'size': len(outlier_topics)
            }
            domain_keywords['miscellaneous'] = ['miscellaneous', 'other']
        
        print(f"Discovered {len(discovered_domains)} semantic domains:")
        for domain_name, info in discovered_domains.items():
            print(f"  {domain_name}: {info['size']} topics, keywords: {info['keywords'][:5]}")
        
        return discovered_domains, domain_keywords
    
    def _fallback_domain_discovery(self, topics, topic_embeddings, n_domains):
        """Fallback domain discovery using K-means clustering"""
        print("Using K-means fallback for domain discovery...")
        
        try:
            n_clusters = min(n_domains, len(topics) // 3, 8)  # Reasonable cluster count
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_assignments = kmeans.fit_predict(topic_embeddings)
            
            discovered_domains = {}
            domain_keywords = {}
            
            for cluster_id in range(n_clusters):
                cluster_topics = [topics[i] for i, c in enumerate(cluster_assignments) if c == cluster_id]
                if len(cluster_topics) == 0:
                    continue
                    
                keywords = self._extract_keywords_from_topics(cluster_topics)
                domain_name = f"domain_{cluster_id}_{keywords[0] if keywords else 'unknown'}"
                
                cluster_centroid = kmeans.cluster_centers_[cluster_id]
                
                discovered_domains[domain_name] = {
                    'topics': cluster_topics,
                    'keywords': keywords,
                    'centroid': cluster_centroid,
                    'size': len(cluster_topics)
                }
                domain_keywords[domain_name] = keywords
            
            return discovered_domains, domain_keywords
            
        except Exception as e:
            print(f"Fallback clustering failed: {e}, using single domain")
            return {'general_domain': {'topics': list(topics), 'keywords': ['general'], 'centroid': np.mean(topic_embeddings, axis=0), 'size': len(topics)}}, {'general_domain': ['general']}
    
    def _extract_keywords_from_topics(self, topics, top_k=5):
        """Extract keywords from a list of topics"""
        all_words = []
        for topic in topics:
            words = [word.strip().lower() for word in topic.replace('_', ' ').split()]
            words = [w for w in words if len(w) > 2 and w.isalpha()]
            all_words.extend(words)
        
        if not all_words:
            return ['general']
            
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(top_k)]
    
    def _generate_domain_name(self, domain_topics, keywords):
        """Generate semantic domain name using embedding similarity"""
        if not keywords:
            return "general_domain"
            
        # Create candidate domain names
        candidates = [
            f"{keywords[0]}_{keywords[1]}" if len(keywords) > 1 else f"{keywords[0]}_domain",
            f"{keywords[0]}_thematic",
            self._extract_common_theme(domain_topics),
        ]
        
        # Remove duplicates and empty candidates
        candidates = list(set([c for c in candidates if c and len(c) > 0]))
        
        if not candidates:
            return "general_domain"
        
        try:
            # Embed candidates and domain topics
            candidate_embeddings = self.embedding_model.encode(candidates)
            topic_embeddings = self.embedding_model.encode(domain_topics)
            
            # Find candidate most similar to domain centroid
            domain_centroid = np.mean(topic_embeddings, axis=0)
            similarities = cosine_similarity(candidate_embeddings, domain_centroid.reshape(1, -1))
            
            best_candidate_idx = np.argmax(similarities)
            return candidates[best_candidate_idx]
        except:
            return candidates[0]
    
    def _extract_common_theme(self, topics):
        """Extract common thematic elements from topic names"""
        # Extract frequent meaningful words
        all_words = []
        for topic in topics:
            words = [w.lower() for w in topic.replace('_', ' ').split() 
                    if len(w) > 3 and w.isalpha()]
            all_words.extend(words)
        
        # Find most frequent thematic word
        word_counts = Counter(all_words)
        if word_counts:
            theme_word = word_counts.most_common(1)[0][0]
            return f"{theme_word}_thematic"
        return "general_domain"
    
    def _discover_narrative_chains(self, topics, discovered_domains, min_chain_length=3):
        """
        Dynamically discover narrative chains using temporal/causal embeddings.
        
        Args:
            topics: List of topic names
            discovered_domains: Domain information from domain discovery
            min_chain_length: Minimum topics per narrative chain
        
        Returns:
            narrative_chains: Discovered narrative relationships
            causal_relationships: Directed graph of causal relationships
        """
        print("=== DYNAMIC NARRATIVE CHAIN DISCOVERY ===")
        
        if len(topics) < min_chain_length * 2:
            print(f"Too few topics ({len(topics)}) for narrative discovery")
            return {}, nx.DiGraph()
        
        # Step 1: Detect causal/temporal language patterns
        causal_indicators = self._detect_causal_language(topics)
        
        # Step 2: Build narrative embedding space
        narrative_embeddings = self._create_narrative_embeddings(topics, causal_indicators)
        
        # Step 3: Discover chains using graph-based clustering
        narrative_graph = self._build_narrative_graph(topics, narrative_embeddings, discovered_domains)
        
        # Step 4: Extract coherent narrative chains
        narrative_chains = self._extract_narrative_chains(narrative_graph, min_chain_length)
        
        # Step 5: Determine chain directions using embedding trajectories
        directed_chains = self._determine_chain_directions(narrative_chains, narrative_embeddings, topics)
        
        print(f"Discovered {len(directed_chains)} narrative chains:")
        for chain_name, chain_info in directed_chains.items():
            print(f"  {chain_name}: {len(chain_info['topics'])} topics")
        
        return directed_chains, narrative_graph
    
    def _detect_causal_language(self, topics):
        """Detect causal/temporal patterns in topic names using finetuned embeddings"""
        causal_seeds = [
            "causes", "leads to", "results in", "due to", "because of",
            "before", "after", "during", "following", "preceding",
            "prevention", "treatment", "outcome", "consequence", "effect",
            "policy", "regulation", "implementation", "enforcement", "compliance"
        ]
        
        try:
            # Embed causal seeds and topics
            seed_embeddings = self.embedding_model.encode(causal_seeds)
            topic_embeddings = self.embedding_model.encode(topics)
            
            # Calculate similarity to causal patterns
            causal_similarities = cosine_similarity(topic_embeddings, seed_embeddings)
            
            # Topics with high causal similarity
            causal_topics = {}
            for i, topic in enumerate(topics):
                max_similarity = np.max(causal_similarities[i])
                if max_similarity > 0.3:  # Configurable threshold
                    best_seed_idx = np.argmax(causal_similarities[i])
                    causal_topics[topic] = {
                        'causal_type': causal_seeds[best_seed_idx],
                        'strength': max_similarity
                    }
            
            print(f"Found {len(causal_topics)} topics with causal language patterns")
            return causal_topics
        except Exception as e:
            print(f"Error in causal language detection: {e}")
            return {}
    
    def _create_narrative_embeddings(self, topics, causal_indicators):
        """Create specialized embeddings for narrative discovery"""
        try:
            # Base embeddings from finetuned model
            base_embeddings = self.embedding_model.encode(topics)
            
            # Augment with causal context for topics with causal indicators
            narrative_contexts = []
            for topic in topics:
                context = topic
                if topic in causal_indicators:
                    # Add causal context to enhance narrative detection
                    causal_type = causal_indicators[topic]['causal_type']
                    context = f"{topic} {causal_type} narrative flow"
                narrative_contexts.append(context)
            
            # Re-embed with narrative context
            narrative_embeddings = self.embedding_model.encode(narrative_contexts)
            
            return narrative_embeddings
        except Exception as e:
            print(f"Error creating narrative embeddings: {e}")
            return self.embedding_model.encode(topics)
    
    def _build_narrative_graph(self, topics, narrative_embeddings, discovered_domains):
        """Build a graph of narrative relationships between topics"""
        try:
            G = nx.Graph()
            
            # Add nodes
            for i, topic in enumerate(topics):
                G.add_node(i, topic=topic)
            
            # Add edges based on narrative similarity
            similarity_matrix = cosine_similarity(narrative_embeddings)
            
            for i in range(len(topics)):
                for j in range(i+1, len(topics)):
                    # Higher threshold for narrative relationships
                    if similarity_matrix[i][j] > 0.4:
                        # Check if topics are in related domains
                        topic_i_domain = self._find_topic_domain(topics[i], discovered_domains)
                        topic_j_domain = self._find_topic_domain(topics[j], discovered_domains)
                        
                        # Boost similarity if in same or related domains
                        domain_boost = 0.1 if topic_i_domain == topic_j_domain else 0.0
                        final_similarity = similarity_matrix[i][j] + domain_boost
                        
                        if final_similarity > 0.45:
                            G.add_edge(i, j, weight=final_similarity)
            
            return G
        except Exception as e:
            print(f"Error building narrative graph: {e}")
            return nx.Graph()
    
    def _find_topic_domain(self, topic, discovered_domains):
        """Find which domain a topic belongs to"""
        for domain_name, domain_info in discovered_domains.items():
            if topic in domain_info['topics']:
                return domain_name
        return 'unknown'
    
    def _extract_narrative_chains(self, narrative_graph, min_chain_length):
        """Extract coherent narrative chains from the graph"""
        try:
            # Find connected components that could be narrative chains
            connected_components = list(nx.connected_components(narrative_graph))
            
            narrative_chains = {}
            chain_id = 0
            
            for component in connected_components:
                if len(component) >= min_chain_length:
                    # Extract the subgraph for this component
                    subgraph = narrative_graph.subgraph(component)
                    
                    # Try to find a path through the component
                    nodes = list(component)
                    topics = [narrative_graph.nodes[node]['topic'] for node in nodes]
                    
                    chain_name = f"narrative_chain_{chain_id}"
                    narrative_chains[chain_name] = {
                        'topics': topics,
                        'nodes': nodes,
                        'subgraph': subgraph
                    }
                    chain_id += 1
            
            return narrative_chains
        except Exception as e:
            print(f"Error extracting narrative chains: {e}")
            return {}
    
    def _determine_chain_directions(self, narrative_chains, narrative_embeddings, topics):
        """Determine chain directions using embedding trajectories"""
        try:
            directed_chains = {}
            
            for chain_name, chain_info in narrative_chains.items():
                chain_topics = chain_info['topics']
                
                if len(chain_topics) < 2:
                    directed_chains[chain_name] = chain_info
                    continue
                
                # Get embeddings for chain topics
                topic_indices = [topics.index(topic) for topic in chain_topics if topic in topics]
                chain_embeddings = [narrative_embeddings[i] for i in topic_indices]
                
                if len(chain_embeddings) < 2:
                    directed_chains[chain_name] = chain_info
                    continue
                
                # Simple ordering by embedding space position (can be enhanced)
                # For now, we'll use a heuristic based on embedding magnitude
                topic_positions = [(i, np.linalg.norm(emb)) for i, emb in enumerate(chain_embeddings)]
                topic_positions.sort(key=lambda x: x[1])
                
                # Create ordered chain
                ordered_topics = [chain_topics[pos[0]] for pos in topic_positions]
                
                directed_chains[chain_name] = {
                    'topics': ordered_topics,
                    'direction': f"{ordered_topics[0]} → ... → {ordered_topics[-1]}",
                    'length': len(ordered_topics)
                }
            
            return directed_chains
        except Exception as e:
            print(f"Error determining chain directions: {e}")
            return narrative_chains
    
    def _choose_semantic_representative(self, cluster_topics, data):
        """
        Choose the best representative topic for a cluster based on semantic centrality
        and document frequency.
        
        Args:
            cluster_topics: List of topics in the cluster
            data: DataFrame with topic and document information
            
        Returns:
            Best representative topic string
        """
        if len(cluster_topics) == 1:
            return cluster_topics[0]
        
        # Score each topic based on multiple criteria
        candidates = []
        
        for topic in cluster_topics:
            score = 0
            
            # 1. Document frequency (higher is better)
            doc_count = len(data[data['extracted_topic'] == topic])
            score += np.log(doc_count + 1) * 2  # Weight 2x
            
            # 2. Topic name length (shorter/more general is better)
            score -= len(topic) * 0.05
            
            # 3. Avoid topics with special characters or complex constructions
            penalty = (topic.count(' - ') + topic.count(' and ') + 
                      topic.count(' in ') + topic.count(' of ')) * 0.3
            score -= penalty
            
            # 4. Prefer topics without numbers or special characters
            if any(char.isdigit() for char in topic):
                score -= 0.5
            if any(char in topic for char in ['_', '/', '\\']):
                score -= 0.3
                
            # 5. Semantic centrality - embed topics and find most central
            # (This will be calculated separately for efficiency)
            
            candidates.append((topic, score))
        
        # If we have embeddings available, add semantic centrality
        if len(cluster_topics) > 2:
            try:
                topic_embeddings = self.embedding_model.encode(cluster_topics)
                # Calculate centroid
                centroid = np.mean(topic_embeddings, axis=0)
                # Calculate distances to centroid
                distances = [np.linalg.norm(emb - centroid) for emb in topic_embeddings]
                
                # Add centrality bonus (lower distance = higher bonus)
                max_distance = max(distances) if distances else 1
                for i, (topic, score) in enumerate(candidates):
                    centrality_bonus = (max_distance - distances[i]) / max_distance
                    candidates[i] = (topic, score + centrality_bonus)
                    
            except Exception as e:
                print(f"Warning: Could not compute semantic centrality: {e}")
        
        # Return topic with highest score
        best_topic = max(candidates, key=lambda x: x[1])[0]
        return best_topic
    
    def _compute_dynamic_similarity(self, topics, discovered_domains):
        """
        OPTIMIZED: Compute similarity using finetuned embeddings and discovered patterns.
        Uses vectorized operations, caching, and FAISS for performance.
        
        Args:
            topics: List of topic names
            discovered_domains: Dynamically discovered domain information
        
        Returns:
            similarity_matrix: Dynamic similarity matrix
            similarity_explanation: Explanation of similarity components
        """
        print("=== OPTIMIZED DYNAMIC SIMILARITY COMPUTATION ===")
        
        n_topics = len(topics)
        
        # Progress tracking
        with tqdm(total=6, desc="Dynamic Similarity", unit="step") as pbar:
            try:
                # Step 1: Cached embedding computation
                pbar.set_description("Computing embeddings")
                embeddings = self._get_cached_embeddings(topics)
                pbar.update(1)
                
                # Step 2: FAISS index setup for fast neighbor search
                pbar.set_description("Setting up FAISS index")
                self._setup_faiss_index(embeddings)
                pbar.update(1)
                
                # Step 3: Vectorized domain-aware similarity
                pbar.set_description("Computing domain similarity")
                domain_similarity = self._compute_domain_aware_similarity_vectorized(topics, discovered_domains, embeddings)
                pbar.update(1)
                
                # Step 4: FAISS-accelerated semantic field similarity
                pbar.set_description("Computing semantic similarity")
                semantic_similarity = self._compute_semantic_field_similarity_faiss(embeddings)
                pbar.update(1)
                
                # Step 5: Vectorized abstraction similarity
                pbar.set_description("Computing abstraction similarity")
                abstraction_similarity = self._compute_abstraction_similarity_vectorized(topics, embeddings)
                pbar.update(1)
                
                # Step 6: Fast weight learning with K-means
                pbar.set_description("Learning optimal weights")
                optimal_weights = self._learn_similarity_weights_fast(embeddings, domain_similarity, 
                                                               semantic_similarity, abstraction_similarity)
                pbar.update(1)
                
                # Combine with learned weights using sparse operations where beneficial
                dynamic_similarity = self._combine_similarities_optimized(
                    domain_similarity, semantic_similarity, abstraction_similarity, optimal_weights
                )
                
                print(f"Optimal similarity weights: {optimal_weights}")
                
                # Clear caches to free memory
                gc.collect()
                
                return dynamic_similarity, optimal_weights
            except Exception as e:
                print(f"Error in dynamic similarity computation: {e}")
                # Fallback to simple cosine similarity
                embeddings = self._get_cached_embeddings(topics)
                return cosine_similarity(embeddings), {'domain': 0.33, 'semantic': 0.33, 'abstraction': 0.34}
    
    def _get_cached_embeddings(self, topics):
        """Get embeddings with caching for performance"""
        cache_key = tuple(sorted(topics))
        
        if cache_key not in self._embedding_cache:
            embeddings = self.embedding_model.encode(topics, show_progress_bar=False)
            self._embedding_cache[cache_key] = embeddings
            self._cached_topics = topics
        
        return self._embedding_cache[cache_key]
    
    def _setup_faiss_index(self, embeddings):
        """Setup FAISS index for fast similarity search"""
        if self._faiss_index is None or embeddings.shape[0] != self._faiss_index.ntotal:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self._faiss_index.add(normalized_embeddings.astype('float32'))
    
    def _compute_domain_aware_similarity_vectorized(self, topics, discovered_domains, embeddings):
        """OPTIMIZED: Vectorized domain-aware similarity computation"""
        try:
            n_topics = len(topics)
            
            # Vectorized base similarity computation
            base_similarity = cosine_similarity(embeddings)
            
            # Create domain mapping vectorized
            topic_to_domain = {}
            for domain_name, domain_info in discovered_domains.items():
                for topic in domain_info['topics']:
                    topic_to_domain[topic] = domain_name
            
            # Create domain matrix for vectorized operations
            domain_matrix = np.zeros((n_topics, n_topics))
            
            for i, topic_i in enumerate(topics):
                domain_i = topic_to_domain.get(topic_i, 'unknown')
                if domain_i != 'unknown':
                    for j, topic_j in enumerate(topics):
                        domain_j = topic_to_domain.get(topic_j, 'unknown')
                        if domain_i == domain_j:
                            domain_matrix[i, j] = 0.15  # Domain boost
            
            # Apply domain boost vectorized
            domain_similarity = np.clip(base_similarity + domain_matrix, 0.0, 1.0)
            
            return domain_similarity
        except Exception as e:
            print(f"Error in vectorized domain-aware similarity: {e}")
            return cosine_similarity(embeddings)
    
    def _compute_semantic_field_similarity_faiss(self, embeddings):
        """OPTIMIZED: FAISS-accelerated semantic field similarity"""
        try:
            n_topics = len(embeddings)
            
            # Base similarity using FAISS (already computed during setup)
            base_similarity = cosine_similarity(embeddings)
            
            # Fast k-NN search using FAISS
            k = min(10, n_topics // 2) if n_topics > 10 else n_topics - 1
            if k <= 0:
                return base_similarity
            
            # Query FAISS for k nearest neighbors for all points at once
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities, indices = self._faiss_index.search(normalized_embeddings.astype('float32'), k + 1)
            
            # Vectorized neighborhood enhancement
            enhanced_similarity = base_similarity.copy()
            
            # Apply smart thresholding - only enhance similarities above threshold
            threshold = 0.3
            
            for i in range(n_topics):
                neighbor_indices = indices[i][1:]  # Exclude self (first result)
                neighbor_sims = similarities[i][1:]
                
                # Only process neighbors above threshold
                valid_neighbors = neighbor_indices[neighbor_sims > threshold]
                
                if len(valid_neighbors) > 1:
                    # Vectorized transitive boost computation
                    for j_idx, j in enumerate(valid_neighbors[:-1]):
                        for k_idx in range(j_idx + 1, len(valid_neighbors)):
                            k_neighbor = valid_neighbors[k_idx]
                            if j < n_topics and k_neighbor < n_topics:
                                # Transitive similarity boost (reduced for performance)
                                boost = 0.03 * base_similarity[i, j] * base_similarity[i, k_neighbor]
                                enhanced_similarity[i, k_neighbor] = min(1.0, enhanced_similarity[i, k_neighbor] + boost)
            
            return enhanced_similarity
        except Exception as e:
            print(f"Error in FAISS semantic field similarity: {e}")
            return cosine_similarity(embeddings)
    
    def _compute_abstraction_similarity_vectorized(self, topics, embeddings):
        """OPTIMIZED: Vectorized abstraction similarity computation"""
        try:
            n_topics = len(topics)
            
            # Vectorized abstraction level computation
            abstraction_levels = np.array([
                1.0 / (1.0 + 0.1 * len(topic.replace('_', ' ').split()) + 
                       0.01 * (sum(len(w) for w in topic.replace('_', ' ').split()) / 
                               max(1, len(topic.replace('_', ' ').split()))))
                for topic in topics
            ])
            
            # Vectorized base similarity
            base_similarity = cosine_similarity(embeddings)
            
            # Vectorized abstraction penalty computation
            level_diff_matrix = np.abs(abstraction_levels[:, np.newaxis] - abstraction_levels[np.newaxis, :])
            abstraction_penalty = 0.1 * level_diff_matrix
            
            # Apply penalty vectorized with smart thresholding
            abstraction_similarity = np.maximum(0.0, base_similarity - abstraction_penalty)
            
            # Set diagonal to 1.0
            np.fill_diagonal(abstraction_similarity, 1.0)
            
            return abstraction_similarity
        except Exception as e:
            print(f"Error in vectorized abstraction similarity: {e}")
            return cosine_similarity(embeddings)
    
    def _learn_similarity_weights_fast(self, embeddings, *similarity_matrices):
        """OPTIMIZED: Fast weight learning using K-means instead of spectral clustering"""
        try:
            best_weights = None
            best_score = -1
            
            # Reduced weight combinations for speed
            weight_combinations = [
                {'domain': 0.5, 'semantic': 0.3, 'abstraction': 0.2},
                {'domain': 0.4, 'semantic': 0.4, 'abstraction': 0.2},
                {'domain': 0.3, 'semantic': 0.4, 'abstraction': 0.3}
            ]
            
            # Early exit for small datasets
            if len(embeddings) < 10:
                return weight_combinations[0]
            
            for weights in weight_combinations:
                try:
                    # Combine similarities with these weights (vectorized)
                    combined = (weights['domain'] * similarity_matrices[0] + 
                               weights['semantic'] * similarity_matrices[1] +
                               weights['abstraction'] * similarity_matrices[2])
                    
                    # Fast K-means clustering instead of spectral
                    n_clusters = min(6, max(2, len(embeddings) // 6))  # Reduced clusters
                    
                    # Use K-means directly on embeddings for speed
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)  # Reduced n_init
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    # Fast silhouette score computation with sampling
                    if len(set(cluster_labels)) > 1:
                        # Sample for large datasets to speed up silhouette computation
                        if len(embeddings) > 100:
                            sample_indices = np.random.choice(len(embeddings), min(100, len(embeddings)), replace=False)
                            score = silhouette_score(embeddings[sample_indices], cluster_labels[sample_indices])
                        else:
                            score = silhouette_score(embeddings, cluster_labels)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights
                except Exception as inner_e:
                    continue
            
            return best_weights or weight_combinations[0]
        except Exception as e:
            print(f"Error in fast weight learning: {e}")
            return {'domain': 0.4, 'semantic': 0.4, 'abstraction': 0.2}
    
    def _combine_similarities_optimized(self, domain_sim, semantic_sim, abstraction_sim, weights):
        """OPTIMIZED: Memory-efficient similarity combination"""
        try:
            # Use in-place operations where possible to save memory
            result = weights['domain'] * domain_sim
            result += weights['semantic'] * semantic_sim
            result += weights['abstraction'] * abstraction_sim
            
            # Apply smart thresholding to create sparse structure if beneficial
            threshold = 0.1
            result[result < threshold] = 0.0
            
            return result
        except Exception as e:
            print(f"Error in optimized similarity combination: {e}")
            # Fallback to simple combination
            return (weights['domain'] * domain_sim + 
                    weights['semantic'] * semantic_sim +
                    weights['abstraction'] * abstraction_sim)
    
    def _consolidate_hierarchical_multilevel_dynamic(self, data, max_levels=3, 
                                                   level_thresholds=None, 
                                                   n_domains=6, min_domain_size=3,
                                                   min_chain_length=3):
        """
        Dynamic hierarchical consolidation using BERTopic and finetuned embeddings.
        Replaces hardcoded patterns with discovered semantic structures.
        
        Args:
            data: DataFrame with 'extracted_topic' and 'text_segment' columns
            max_levels: Maximum number of hierarchy levels to create
            level_thresholds: List of similarity thresholds for each level
            n_domains: Target number of domains to discover
            min_domain_size: Minimum topics per domain
            min_chain_length: Minimum topics per narrative chain
            
        Returns:
            data: DataFrame with multiple level columns and dynamic features
            consolidation_info: Dictionary with hierarchy and discovery results
        """
        print("=== DYNAMIC HIERARCHICAL CONSOLIDATION ==")
        
        if level_thresholds is None:
            level_thresholds = [0.8, 0.65, 0.5]
            
        unique_topics = data['extracted_topic'].unique()
        print(f"Building dynamic {max_levels}-level hierarchy from {len(unique_topics)} topics")
        
        # Overall progress tracking
        total_steps = 8
        overall_pbar = tqdm(total=total_steps, desc="Dynamic Consolidation", position=0)
        
        # Step 1: Dynamic domain discovery using BERTopic
        overall_pbar.set_description("Discovering semantic domains")
        discovered_domains, domain_keywords = self._discover_domains_dynamically(
            unique_topics, n_domains, min_domain_size
        )
        overall_pbar.update(1)
        
        # Step 2: Filter to domain-relevant topics using discovered patterns
        overall_pbar.set_description("Filtering domain-relevant topics")
        all_domain_keywords = []
        for domain_name, keywords in domain_keywords.items():
            all_domain_keywords.extend(keywords)
        
        filtered_topics = self._filter_domain_relevant_topics(
            unique_topics, 
            domain_keywords=all_domain_keywords,
            similarity_threshold=0.3
        )
        
        # Filter data to only include relevant topics
        data = data[data['extracted_topic'].isin(filtered_topics)].copy()
        print(f"Filtered dataset from {len(unique_topics)} to {len(filtered_topics)} topics")
        overall_pbar.update(1)
        
        # Step 3: Discover narrative chains in filtered data
        overall_pbar.set_description("Discovering narrative chains")
        narrative_chains, narrative_graph = self._discover_narrative_chains(
            filtered_topics, discovered_domains, min_chain_length
        )
        overall_pbar.update(1)
        
        # Step 4: Compute dynamic similarity using all discovered patterns (OPTIMIZED)
        overall_pbar.set_description("Computing optimized similarity")
        start_time = time.time()
        dynamic_similarity, optimal_weights = self._compute_dynamic_similarity(
            filtered_topics, discovered_domains
        )
        step4_time = time.time() - start_time
        print(f"Step 4 completed in {step4_time:.2f} seconds")
        overall_pbar.update(1)
        
        # Step 5: Build dynamic hierarchy using learned similarity
        overall_pbar.set_description("Building dynamic hierarchy")
        current_topics = list(filtered_topics)
        hierarchy = {'level_0': {topic: [topic] for topic in current_topics}}
        level_stats = {'level_0': len(current_topics)}
        
        for level in range(1, max_levels + 1):
            threshold = level_thresholds[level-1] if level-1 < len(level_thresholds) else 0.5
            print(f"\\nLevel {level}: Using dynamic similarity threshold {threshold}")
            
            if len(current_topics) <= 1:
                print(f"Only {len(current_topics)} topic(s) remaining, stopping hierarchy building")
                break
            
            # Use dynamic similarity for clustering
            topic_indices = [i for i, topic in enumerate(filtered_topics) if topic in current_topics]
            current_similarity = dynamic_similarity[np.ix_(topic_indices, topic_indices)]
            
            # Convert to distance matrix for clustering
            distance_matrix = 1 - current_similarity
            np.fill_diagonal(distance_matrix, 0)
            
            # Hierarchical clustering with dynamic similarity
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1-threshold,
                metric='precomputed',
                linkage='average'
            )
            
            clusters = clustering.fit_predict(distance_matrix)
            n_clusters = len(np.unique(clusters))
            
            print(f"Created {n_clusters} dynamic clusters at level {level}")
            
            # Create consolidated groups for this level
            level_groups = {}
            for cluster_id in np.unique(clusters):
                cluster_topics = [current_topics[i] for i in range(len(current_topics)) 
                                if clusters[i] == cluster_id]
                
                if len(cluster_topics) == 1:
                    representative = cluster_topics[0]
                else:
                    # Choose representative using discovered domain knowledge
                    representative = self._choose_dynamic_representative(
                        cluster_topics, data, discovered_domains, narrative_chains
                    )
                
                level_groups[representative] = cluster_topics
            
            hierarchy[f'level_{level}'] = level_groups
            level_stats[f'level_{level}'] = len(level_groups)
            current_topics = list(level_groups.keys())
            
            print(f"Level {level}: {len(level_groups)} dynamic groups")
            
            # Stop if no further consolidation
            if len(level_groups) == len(current_topics):
                print(f"No consolidation at level {level}, stopping")
                break
        
        # Step 6: Apply hierarchy to data with dynamic features
        overall_pbar.set_description("Applying hierarchy to data")
        overall_pbar.update(1)
        
        for level in range(1, max_levels + 1):
            level_key = f'level_{level}'
            if level_key not in hierarchy:
                break
                
            # Create mapping from original topics to this level's representatives
            level_map = {}
            for representative, members in hierarchy[level_key].items():
                for member in members:
                    level_map[member] = representative
            
            # Apply mapping to data
            data[f'topic_{level_key}'] = data['extracted_topic'].map(level_map)
        
        # Step 7: Add dynamic domain and narrative chain information
        overall_pbar.set_description("Adding semantic features")
        
        # Add domain assignments
        topic_to_domain = {}
        for domain_name, domain_info in discovered_domains.items():
            for topic in domain_info['topics']:
                topic_to_domain[topic] = domain_name
        
        data['discovered_domain'] = data['extracted_topic'].map(topic_to_domain)
        
        # Add narrative chain assignments
        topic_to_chain = {}
        for chain_name, chain_info in narrative_chains.items():
            for topic in chain_info['topics']:
                topic_to_chain[topic] = chain_name
        
        data['narrative_chain'] = data['extracted_topic'].map(topic_to_chain).fillna('standalone')
        
        # Step 8: Generate comprehensive results
        overall_pbar.set_description("Generating results")
        overall_pbar.update(1)
        consolidation_info = {
            'hierarchy': hierarchy,
            'level_stats': level_stats,
            'method': 'hierarchical_multilevel_dynamic',
            'total_levels': len([k for k in hierarchy.keys() if k.startswith('level_')]),
            'compression_ratios': {},
            'discovered_domains': discovered_domains,
            'narrative_chains': narrative_chains,
            'dynamic_similarity_weights': optimal_weights,
            'domain_keywords': domain_keywords
        }
        
        # Calculate compression ratios
        base_count = level_stats['level_0']
        for level, count in level_stats.items():
            if level != 'level_0':
                ratio = base_count / count if count > 0 else float('inf')
                consolidation_info['compression_ratios'][level] = ratio
        
        # Show results
        overall_pbar.set_description("Complete!")
        overall_pbar.update(1)
        overall_pbar.close()
        
        print(f"\\n=== DYNAMIC HIERARCHY SUMMARY ===")
        print(f"Original topics: {base_count}")
        print(f"Discovered domains: {len(discovered_domains)}")
        print(f"Narrative chains: {len(narrative_chains)}")
        print(f"Optimal similarity weights: {optimal_weights}")
        
        for level in sorted(level_stats.keys()):
            if level != 'level_0':
                count = level_stats[level]
                ratio = consolidation_info['compression_ratios'][level]
                print(f"{level}: {count} topics (compression: {ratio:.1f}x)")
        
        return data, consolidation_info
    
    def _choose_dynamic_representative(self, cluster_topics, data, discovered_domains, narrative_chains):
        """
        Choose representative using discovered domain and narrative knowledge.
        """
        if len(cluster_topics) == 1:
            return cluster_topics[0]
        
        candidates = []
        
        for topic in cluster_topics:
            score = 0
            
            # 1. Document frequency (higher is better)
            doc_count = len(data[data['extracted_topic'] == topic])
            score += np.log(doc_count + 1) * 2
            
            # 2. Domain centrality bonus
            for domain_name, domain_info in discovered_domains.items():
                if topic in domain_info['topics']:
                    # Bonus for being in discovered domain
                    score += 1.0
                    # Extra bonus if topic appears in domain keywords
                    if any(keyword in topic.lower() for keyword in domain_info['keywords']):
                        score += 0.5
                    break
            
            # 3. Narrative chain bonus
            for chain_name, chain_info in narrative_chains.items():
                if topic in chain_info['topics']:
                    # Bonus for being in narrative chain
                    score += 0.5
                    break
            
            # 4. Standard criteria (topic length, complexity)
            score -= len(topic) * 0.05
            penalty = (topic.count(' - ') + topic.count(' and ') + 
                      topic.count(' in ') + topic.count(' of ')) * 0.3
            score -= penalty
            
            candidates.append((topic, score))
        
        # Return topic with highest dynamic score
        best_topic = max(candidates, key=lambda x: x[1])[0]
        return best_topic
    
    def _consolidate_hierarchical_multilevel(self, data, max_levels=3, 
                                           level_thresholds=None, domain_keywords=None):
        """
        Multi-level hierarchical consolidation preserving semantic granularity.
        Creates topic taxonomy instead of flat clustering.
        
        Args:
            data: DataFrame with 'extracted_topic' and 'text_segment' columns
            max_levels: Maximum number of hierarchy levels to create
            level_thresholds: List of similarity thresholds for each level (high to low)
            domain_keywords: Keywords for domain filtering
            
        Returns:
            data: DataFrame with multiple level columns
            hierarchy_info: Dictionary with hierarchy structure and statistics
        """
        print("=== HIERARCHICAL MULTI-LEVEL CONSOLIDATION ===")
        
        if level_thresholds is None:
            level_thresholds = [0.8, 0.65, 0.5]  # Strict to loose
            
        unique_topics = data['extracted_topic'].unique()
        print(f"Building {max_levels}-level hierarchy from {len(unique_topics)} topics")
        
        # Step 1: Domain filtering (critical improvement)
        if domain_keywords is not None:
            print("Applying domain filtering...")
            filtered_topics = self._filter_domain_relevant_topics(unique_topics, domain_keywords)
            
            # Filter data to only include relevant topics
            data = data[data['extracted_topic'].isin(filtered_topics)].copy()
            print(f"Filtered dataset from {len(unique_topics)} to {len(filtered_topics)} topics")
            unique_topics = filtered_topics
        
        # Step 2: Build hierarchy bottom-up
        current_topics = list(unique_topics)
        hierarchy = {'level_0': {topic: [topic] for topic in current_topics}}
        level_stats = {'level_0': len(current_topics)}
        
        for level in range(1, max_levels + 1):
            threshold = level_thresholds[level-1] if level-1 < len(level_thresholds) else 0.5
            print(f"\nLevel {level}: Using similarity threshold {threshold}")
            
            if len(current_topics) <= 1:
                print(f"Only {len(current_topics)} topic(s) remaining, stopping hierarchy building")
                break
            
            # Compute embeddings for current level topics
            print(f"Computing embeddings for {len(current_topics)} topics...")
            topic_embeddings = self.embedding_model.encode(current_topics)
            
            # Hierarchical clustering at this level
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1-threshold,
                metric='cosine',
                linkage='average'
            )
            
            clusters = clustering.fit_predict(topic_embeddings)
            n_clusters = len(np.unique(clusters))
            
            print(f"Created {n_clusters} clusters at level {level}")
            
            # Create consolidated groups for this level
            level_groups = {}
            for cluster_id in np.unique(clusters):
                cluster_topics = [current_topics[i] for i in range(len(current_topics)) 
                                if clusters[i] == cluster_id]
                
                if len(cluster_topics) == 1:
                    representative = cluster_topics[0]
                else:
                    # Choose representative based on semantic centrality and other criteria
                    representative = self._choose_semantic_representative(cluster_topics, data)
                
                level_groups[representative] = cluster_topics
            
            hierarchy[f'level_{level}'] = level_groups
            level_stats[f'level_{level}'] = len(level_groups)
            current_topics = list(level_groups.keys())
            
            print(f"Level {level}: {len(level_groups)} consolidated groups")
            
            # Show example groups
            if len(level_groups) <= 10:  # Show all if few groups
                for rep, members in level_groups.items():
                    if len(members) > 1:
                        print(f"  '{rep}' ← {members[:3]}{'...' if len(members) > 3 else ''}")
            
            # Stop if no further consolidation happened
            if len(level_groups) == len(current_topics):
                print(f"No consolidation at level {level}, stopping")
                break
        
        # Step 3: Apply multi-level consolidation to data
        print(f"\nApplying hierarchy to {len(data)} documents...")
        
        for level in range(1, max_levels + 1):
            level_key = f'level_{level}'
            if level_key not in hierarchy:
                break
                
            # Create mapping from original topics to this level's representatives
            level_map = {}
            for representative, members in hierarchy[level_key].items():
                for member in members:
                    level_map[member] = representative
            
            # Apply mapping to data
            data[f'topic_{level_key}'] = data['extracted_topic'].map(level_map)
        
        # Step 4: Generate hierarchy statistics
        hierarchy_info = {
            'hierarchy': hierarchy,
            'level_stats': level_stats,
            'method': 'hierarchical_multilevel',
            'total_levels': len([k for k in hierarchy.keys() if k.startswith('level_')]),
            'compression_ratios': {}
        }
        
        # Calculate compression ratios
        base_count = level_stats['level_0']
        for level, count in level_stats.items():
            if level != 'level_0':
                ratio = base_count / count if count > 0 else float('inf')
                hierarchy_info['compression_ratios'][level] = ratio
        
        print(f"\n=== HIERARCHY SUMMARY ===")
        print(f"Original topics: {base_count}")
        for level in sorted(level_stats.keys()):
            if level != 'level_0':
                count = level_stats[level]
                ratio = hierarchy_info['compression_ratios'][level]
                print(f"{level}: {count} topics (compression: {ratio:.1f}x)")
        
        return data, hierarchy_info
        
    def consolidate_topics(self, data, method='hybrid', **kwargs):
        """
        Main method to consolidate topics.
        
        Args:
            data: DataFrame with 'extracted_topic' and 'text_segment' columns
            method: Consolidation method:
                   - 'name_similarity': Cluster by topic name similarity
                   - 'content_similarity': Cluster by document content similarity  
                   - 'bertopic': Use BERTopic for meta-topic discovery
                   - 'hybrid': Original hybrid approach with domain coherence
                   - 'hierarchical': Multi-level hierarchical consolidation
                   - 'hierarchical_dynamic': NEW - Fully dynamic discovery with BERTopic
                   - 'hybrid_with_stance': Hierarchical + separate stance detection
                   - 'dynamic_with_stance': NEW - Dynamic discovery + separate stance detection
            **kwargs: Additional arguments passed to specific methods
        
        Returns:
            data: DataFrame with consolidation columns
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
        elif method == 'hierarchical':
            return self._consolidate_hierarchical_multilevel(data, **kwargs)
        elif method == 'hierarchical_dynamic':
            # NEW: Fully dynamic hierarchical consolidation with BERTopic discovery
            return self._consolidate_hierarchical_multilevel_dynamic(data, **kwargs)
        elif method == 'hybrid_with_stance':
            # First perform separate stance detection
            print("=== HYBRID WITH SEPARATE STANCE DETECTION ===")
            
            # Extract stance-specific kwargs
            stance_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['stance_keywords', 'confidence_threshold']}
            
            # Extract hierarchy-specific kwargs
            hierarchy_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['max_levels', 'level_thresholds', 'domain_keywords']}
            
            data, stance_stats = self.detect_stance_separately(data, **stance_kwargs)
            
            # Then perform hierarchical consolidation  
            data, hierarchy_info = self._consolidate_hierarchical_multilevel(data, **hierarchy_kwargs)
            
            # Combine results
            consolidation_info = hierarchy_info
            consolidation_info['stance_statistics'] = stance_stats
            consolidation_info['method'] = 'hybrid_with_stance'
            
            return data, consolidation_info
        elif method == 'dynamic_with_stance':
            # NEW: Dynamic discovery + separate stance detection
            print("=== DYNAMIC DISCOVERY WITH SEPARATE STANCE DETECTION ===")
            
            # Extract stance-specific kwargs
            stance_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['stance_keywords', 'confidence_threshold']}
            
            # Extract hierarchy-specific kwargs
            hierarchy_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['max_levels', 'level_thresholds', 'n_domains', 
                                     'min_domain_size', 'min_chain_length']}
            
            data, stance_stats = self.detect_stance_separately(data, **stance_kwargs)
            
            # Then perform dynamic hierarchical consolidation  
            data, hierarchy_info = self._consolidate_hierarchical_multilevel_dynamic(data, **hierarchy_kwargs)
            
            # Combine results
            consolidation_info = hierarchy_info
            consolidation_info['stance_statistics'] = stance_stats
            consolidation_info['method'] = 'dynamic_with_stance'
            
            return data, consolidation_info
        else:
            available_methods = ['name_similarity', 'content_similarity', 'bertopic', 
                               'hybrid', 'hierarchical', 'hierarchical_dynamic', 
                               'hybrid_with_stance', 'dynamic_with_stance']
            raise ValueError(f"Unknown method: {method}. Available methods: {available_methods}")
    
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
        
        # 4. Domain coherence (UPDATED - replaces stance compatibility)
        print("Step 4: Computing domain coherence...")
        domain_coherence = np.ones((n_topics, n_topics))  # Default: coherent
        
        # Check if topics belong to same semantic domain using multiple signals
        for i, topic_i in enumerate(unique_topics):
            for j, topic_j in enumerate(unique_topics):
                if i != j:
                    # Use word overlap and semantic patterns for domain coherence
                    words_i = set(topic_i.lower().replace('_', ' ').split())
                    words_j = set(topic_j.lower().replace('_', ' ').split())
                    
                    if len(words_i) > 0 and len(words_j) > 0:
                        # Calculate word overlap (Jaccard similarity)
                        word_overlap = len(words_i & words_j) / len(words_i | words_j)
                        
                        # Semantic domain patterns - topics in same domain should cluster
                        domain_patterns = {
                            'medical': ['medical', 'health', 'doctor', 'clinical', 'treatment', 'procedure', 'surgery', 'medication'],
                            'legal': ['legal', 'law', 'court', 'constitutional', 'legislation', 'policy', 'regulation', 'judicial'],
                            'ethical': ['moral', 'ethical', 'values', 'beliefs', 'religion', 'conscience', 'philosophy'],
                            'social': ['social', 'society', 'cultural', 'community', 'public', 'demographic'],
                            'economic': ['economic', 'financial', 'cost', 'money', 'budget', 'insurance', 'funding'],
                            'political': ['political', 'government', 'policy', 'election', 'vote', 'party', 'congress']
                        }
                        
                        # Check if both topics belong to same domain
                        topic_i_domains = set()
                        topic_j_domains = set()
                        
                        for domain, keywords in domain_patterns.items():
                            if any(keyword in topic_i.lower() for keyword in keywords):
                                topic_i_domains.add(domain)
                            if any(keyword in topic_j.lower() for keyword in keywords):
                                topic_j_domains.add(domain)
                        
                        # Higher coherence for topics in same domain
                        domain_match = len(topic_i_domains & topic_j_domains) > 0
                        domain_bonus = 0.3 if domain_match else 0.0
                        
                        # Final domain coherence score
                        domain_coherence[i, j] = word_overlap + domain_bonus
        
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
        
        # 6. Combine similarities (UPDATED with domain coherence instead of stance)
        print("Step 6: Combining all signals...")
        combined_similarity = (
            0.3 * name_similarity + 
            0.4 * content_similarity + 
            0.15 * structural_similarity +
            0.15 * domain_coherence  # Domain coherence instead of stance compatibility
        )
        
        # 7. Graph-based clustering
        print("Step 7: Graph-based consolidation with domain awareness...")
        
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
        narrative_chain_groups = 0
        domain_coherent_groups = 0
        
        for group in consolidated_groups:
            members = group['members']
            if len(members) > 1:
                # Check domain coherence - topics should be in same semantic domain
                domain_keywords = ['medical', 'legal', 'ethical', 'social', 'economic', 'political']
                group_domains = set()
                for topic in members:
                    topic_lower = topic.lower()
                    for domain in domain_keywords:
                        if domain in topic_lower:
                            group_domains.add(domain)
                
                # If group has dominant domain, it's coherent
                if len(group_domains) <= 2:  # Allow some overlap
                    domain_coherent_groups += 1
                
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
                        # Add document count info instead of stance
                        doc_count = len(data[data['extracted_topic'] == member])
                        print(f"  - {member} [docs: {doc_count}]")
        
        print(f"\n✅ Domain-coherent groups: {domain_coherent_groups}/{len(consolidated_groups)}")
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
            'domain_coherence_matrix': domain_coherence,
            'narrative_relationships_matrix': narrative_relationships,
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