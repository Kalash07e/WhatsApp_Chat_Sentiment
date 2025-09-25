"""
Topic modeling utilities using LDA.
"""
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from ..config import config


class TopicModeler:
    """Handles topic modeling using Latent Dirichlet Allocation."""
    
    def __init__(self, n_topics: int = None):
        self.n_topics = n_topics or config.app.TOPIC_MODEL_COMPONENTS
        self.vectorizer = None
        self.lda_model = None
    
    def extract_topics(self, messages: List[str], min_messages: int = None) -> Optional[List[str]]:
        """
        Extract topics from a list of messages.
        
        Args:
            messages: List of message texts
            min_messages: Minimum number of messages required
            
        Returns:
            List of topic descriptions or None if not enough data
        """
        min_messages = min_messages or config.app.MIN_TOPIC_MESSAGES
        
        if len(messages) < min_messages:
            return None
        
        # Clean and filter messages
        cleaned_messages = [msg.strip() for msg in messages if msg and msg.strip()]
        
        if len(cleaned_messages) < min_messages:
            return None
        
        try:
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_df=0.9,
                min_df=2,
                stop_words='english',
                max_features=1000
            )
            
            # Transform messages to document-term matrix
            dtm = self.vectorizer.fit_transform(cleaned_messages)
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=10
            )
            
            self.lda_model.fit(dtm)
            
            # Extract topic descriptions
            return self._extract_topic_descriptions()
            
        except ValueError as e:
            # Not enough unique terms or other vectorization issues
            return None
        except Exception as e:
            # Other unexpected errors
            return None
    
    def _extract_topic_descriptions(self, top_words: int = 5) -> List[str]:
        """Extract readable topic descriptions from the model."""
        if not self.lda_model or not self.vectorizer:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get top words for this topic
            top_word_indices = topic.argsort()[-top_words:][::-1]
            top_words_list = [feature_names[i] for i in top_word_indices]
            
            topic_description = f"Topic {topic_idx + 1}: {', '.join(top_words_list)}"
            topics.append(topic_description)
        
        return topics
    
    def get_document_topics(self, messages: List[str]) -> Optional[List[Tuple[int, float]]]:
        """
        Get topic distribution for each document.
        
        Args:
            messages: List of message texts
            
        Returns:
            List of (topic_id, probability) tuples for each message
        """
        if not self.lda_model or not self.vectorizer:
            return None
        
        try:
            dtm = self.vectorizer.transform(messages)
            doc_topic_probs = self.lda_model.transform(dtm)
            
            # For each document, find the dominant topic
            dominant_topics = []
            for doc_probs in doc_topic_probs:
                dominant_topic_id = doc_probs.argmax()
                max_prob = doc_probs[dominant_topic_id]
                dominant_topics.append((dominant_topic_id, max_prob))
            
            return dominant_topics
            
        except Exception:
            return None
