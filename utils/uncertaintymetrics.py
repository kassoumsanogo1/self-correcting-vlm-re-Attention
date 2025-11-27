

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


from utils.uncertaintymetrics import UncertaintyMetrics

@dataclass
class UncertaintyMetrics:
    """Container for uncertainty measurements"""
    overall: float
    token_level: float
    attention_level: float
    semantic_level: float
    claim_level: float
    uncertain_spans: List[Dict]



class UncertaintyAnalyzer:
    """
    Multi-dimensional uncertainty quantification for VLM responses.
    
    Combines 4 complementary metrics:
    1. Token-level entropy
    2. Attention dispersion
    3. Semantic consistency (via sampling)
    4. Claim-level confidence (linguistic markers)
    """
    
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Hedge words indicating uncertainty
        self.hedge_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could',
            'approximately', 'roughly', 'around', 'about',
            'seems', 'appears', 'likely', 'probably'
        }
    
    def compute_uncertainty(
        self,
        image: Image.Image,
        question: str,
        response: str,
        logits_history: Optional[torch.Tensor] = None
    ) -> UncertaintyMetrics:
        """
        Compute comprehensive uncertainty score
        
        Args:
            image: PIL Image
            question: Input question
            response: Model's response
            logits_history: Tensor of shape [seq_len, vocab_size]
        
        Returns:
            UncertaintyMetrics object with all scores
        """
        
        # Metric 1: Token-level entropy
        if logits_history is not None:
            token_uncertainty = self._compute_token_entropy(logits_history)
        else:
            token_uncertainty = 0.5  # Default if logits unavailable
        
        # Metric 2: Attention dispersion
        attention_uncertainty = self._compute_attention_dispersion(image, question)
        
        # Metric 3: Semantic consistency via sampling
        semantic_uncertainty = self._compute_semantic_consistency(
            image, question, num_samples=3
        )
        
        # Metric 4: Claim-level confidence
        claim_uncertainty = self._compute_claim_confidence(response)
        
        # Weighted combination (empirically optimized)
        overall_uncertainty = (
            0.30 * token_uncertainty +
            0.25 * attention_uncertainty +
            0.25 * semantic_uncertainty +
            0.20 * claim_uncertainty
        )
        
        # Identify specific uncertain spans
        uncertain_spans = self._identify_uncertain_spans(
            response, logits_history if logits_history is not None else None
        )
        
        return UncertaintyMetrics(
            overall=overall_uncertainty,
            token_level=token_uncertainty,
            attention_level=attention_uncertainty,
            semantic_level=semantic_uncertainty,
            claim_level=claim_uncertainty,
            uncertain_spans=uncertain_spans
        )
    
    def _compute_token_entropy(self, logits_history: torch.Tensor) -> float:
        """
        Calculate average entropy of token probability distributions
        H = -Σ p(x) log p(x)
        
        High entropy = model uncertain about next token
        """
        entropies = []
        
        for logits in logits_history:
            probs = F.softmax(logits, dim=-1)
            # Shannon entropy (clip for numerical stability)
            entropy = -torch.sum(
                probs * torch.log(probs.clamp(min=1e-10)),
                dim=-1
            )
            entropies.append(entropy.item())
        
        # Normalize by max possible entropy (log vocab_size)
        vocab_size = logits_history.shape[-1]
        max_entropy = np.log(vocab_size)
        normalized_entropy = np.mean(entropies) / max_entropy
        
        return float(normalized_entropy)
    
    def _compute_attention_dispersion(
        self,
        image: Image.Image,
        question: str
    ) -> float:
        """
        Measure dispersion of cross-modal attention weights
        
        High dispersion = model not focused on specific regions
        → likely uncertain or hallucinating
        """
        try:
            inputs = self.processor(
                text=question,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # Extract cross-attention from last layer
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                cross_attention = outputs.cross_attentions[-1]
                # Shape: [batch, num_heads, seq_len, num_patches]
                
                # Average over heads and sequence
                attention_weights = cross_attention.mean(dim=1).mean(dim=1)
                
                # Calculate entropy of attention distribution
                attention_weights = attention_weights.flatten()
                attention_probs = F.softmax(attention_weights, dim=-1)
                attention_entropy = -torch.sum(
                    attention_probs * torch.log(attention_probs.clamp(min=1e-10))
                ).item()
                
                # Normalize by max entropy
                max_entropy = np.log(len(attention_weights))
                normalized = attention_entropy / max_entropy
                
                return float(normalized)
            else:
                return 0.5  # Default if attention not available
                
        except Exception as e:
            print(f"Warning: Attention extraction failed: {e}")
            return 0.5
    
    def _compute_semantic_consistency(
        self,
        image: Image.Image,
        question: str,
        num_samples: int = 3
    ) -> float:
        """
        Generate multiple responses with sampling and measure variance
        
        High variance = model inconsistent → uncertain
        """
        responses = []
        
        try:
            inputs = self.processor(
                text=question,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                response = self.processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                responses.append(response)
            
            # Measure pairwise similarity
            similarities = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    sim = self._simple_text_similarity(responses[i], responses[j])
                    similarities.append(sim)
            
            # Low similarity = high uncertainty
            avg_similarity = np.mean(similarities) if similarities else 0.5
            uncertainty = 1.0 - avg_similarity
            
            return float(uncertainty)
            
        except Exception as e:
            print(f"Warning: Semantic consistency computation failed: {e}")
            return 0.5
    
    def _compute_claim_confidence(self, response: str) -> float:
        """
        Analyze linguistic markers of uncertainty in response
        
        - Hedge words → low confidence
        - Specific numbers/details → high confidence
        - Vague language → low confidence
        """
        response_lower = response.lower()
        words = response_lower.split()
        
        # Count hedge words
        hedge_count = sum(1 for word in words if word in self.hedge_words)
        hedge_ratio = hedge_count / max(len(words), 1)
        
        # Check for vague quantifiers
        vague_quantifiers = {'some', 'several', 'few', 'many', 'various'}
        vague_count = sum(1 for word in words if word in vague_quantifiers)
        vague_ratio = vague_count / max(len(words), 1)
        
        # Check for specific numbers (positive signal)
        import re
        number_pattern = r'\b\d+\b'
        specific_numbers = len(re.findall(number_pattern, response))
        specificity_bonus = min(specific_numbers * 0.1, 0.3)
        
        # Combine signals
        uncertainty = (
            0.5 * hedge_ratio +
            0.3 * vague_ratio -
            0.2 * specificity_bonus
        )
        
        return float(np.clip(uncertainty, 0.0, 1.0))
    
    def _identify_uncertain_spans(
        self,
        response: str,
        logits_history: Optional[torch.Tensor]
    ) -> List[Dict]:
        """
        Identify specific text spans with high uncertainty
        """
        uncertain_spans = []
        
        # Method 1: Hedge-based detection
        words = response.split()
        for i, word in enumerate(words):
            if word.lower() in self.hedge_words:
                uncertain_spans.append({
                    "text": word,
                    "position": i,
                    "type": "hedge_word",
                    "confidence": 0.3
                })
        
        # Method 2: Logits-based detection (if available)
        if logits_history is not None:
            try:
                tokens = self.processor.tokenizer.tokenize(response)
                for i, logits in enumerate(logits_history[:len(tokens)]):
                    probs = F.softmax(logits, dim=-1)
                    top_prob = probs.max().item()
                    
                    if top_prob < 0.5:  # Low confidence threshold
                        uncertain_spans.append({
                            "text": tokens[i] if i < len(tokens) else "",
                            "position": i,
                            "type": "low_probability",
                            "confidence": top_prob
                        })
            except Exception as e:
                pass  # Fallback to hedge-based only
        
        return uncertain_spans
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity (Jaccard)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

