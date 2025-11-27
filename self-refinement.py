"""
Self-Correcting Vision-Language Models via Uncertainty-Guided Visual Re-Attention

Main implementation of the self-refinement system for reducing hallucinations
in open-source VLMs through iterative uncertainty-guided verification.

Author: Research Implementation
License: MIT
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty measurements"""
    overall: float
    token_level: float
    attention_level: float
    semantic_level: float
    claim_level: float
    uncertain_spans: List[Dict]


@dataclass
class RefinementResult:
    """Container for refinement output"""
    final_response: str
    final_uncertainty: float
    iterations: int
    history: List[Dict]
    improvement: float


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
      """
      try:
          messages = [
              {
                  "role": "user",
                  "content": [
                      {"type": "image", "image": image},
                      {"type": "text", "text": question},
                  ],
              }
          ]
          
          inputs = self.processor.apply_chat_template(
              messages,
              tokenize=True,
              add_generation_prompt=True,
              return_dict=True,
              return_tensors="pt"
          ).to(self.device)
          
          with torch.no_grad():
              outputs = self.model(**inputs, output_attentions=True)
          
          # FIX: Vérifier si attentions existe et n'est pas None
          if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
              attention = outputs.attentions[-1]
              
              # Vérifier que attention n'est pas None
              if attention is not None:
                  # Average over heads and sequence
                  attention_weights = attention.mean(dim=1).mean(dim=1)
                  
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
          
          # Default si attention non disponible
          return 0.5
                  
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            for _ in range(num_samples):
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
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


class VisualReAttention:
    """
    Uncertainty-guided visual re-attention mechanism.
    
    Identifies under-explored image regions and generates
    targeted crops for verification.
    """
    
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
    
    def generate_attention_guided_crops(
        self,
        image: Image.Image,
        question: str,
        uncertainty_info: UncertaintyMetrics,
        max_crops: int = 3
    ) -> List[Dict]:
        """
        Generate crops of under-explored regions
        
        Args:
            image: Original image
            question: Question being asked
            uncertainty_info: Uncertainty metrics
            max_crops: Maximum number of crops to generate
        
        Returns:
            List of crop dictionaries with image, bbox, scale
        """
        
        # Extract attention heatmap
        attention_map = self._extract_attention_heatmap(image, question)
        
        if attention_map is None:
            # Fallback: grid-based crops
            return self._generate_grid_crops(image, n=max_crops)
        
        # Identify under-explored regions
        underexplored_regions = self._identify_underexplored_regions(
            attention_map,
            image.size
        )
        
        # Generate multi-scale crops
        crops = []
        for region in underexplored_regions[:max_crops]:
            crop_configs = self._generate_multi_scale_crops(
                image,
                region,
                scales=[1.2, 1.5, 2.0]
            )
            crops.extend(crop_configs)
        
        return crops[:max_crops * 2]  # Limit total crops
    
    def _extract_attention_heatmap(
        self,
        image: Image.Image,
        question: str
    ) -> Optional[np.ndarray]:
        """
        Extract cross-modal attention as 2D heatmap
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            if not hasattr(outputs, 'attentions') or not outputs.attentions:
                return None
            
            # Get last layer attention
            attention = outputs.attentions[-1]
            # Average over heads and tokens
            attention_map = attention.mean(dim=1).squeeze()
            
            # For Qwen3-VL, we need to identify vision tokens
            # This is a simplified approach - adjust based on actual architecture
            if attention_map.dim() == 2:
                attention_map = attention_map.mean(dim=0)
            
            # Try to reshape to 2D spatial layout
            num_patches = attention_map.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size == num_patches:
                attention_2d = attention_map.reshape(grid_size, grid_size)
                
                # Resize to image dimensions
                attention_2d = torch.nn.functional.interpolate(
                    attention_2d.unsqueeze(0).unsqueeze(0),
                    size=(image.height, image.width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                return attention_2d.cpu().numpy()
            else:
                return None
            
        except Exception as e:
            print(f"Warning: Attention extraction failed: {e}")
            return None
    
    def _identify_underexplored_regions(
        self,
        attention_map: np.ndarray,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Find regions with low attention scores
        """
        # Threshold: regions with attention < 20% of max
        threshold = 0.2 * attention_map.max()
        underexplored_mask = attention_map < threshold
        
        # Connected component analysis
        labeled, num_features = ndimage.label(underexplored_mask)
        
        regions = []
        for region_id in range(1, num_features + 1):
            region_mask = labeled == region_id
            
            # Extract bounding box
            rows = np.any(region_mask, axis=1)
            cols = np.any(region_mask, axis=0)
            
            if not rows.any() or not cols.any():
                continue
            
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            
            ymin, ymax = row_indices[0], row_indices[-1]
            xmin, xmax = col_indices[0], col_indices[-1]
            
            # Filter very small regions (< 5% of image)
            area = (ymax - ymin) * (xmax - xmin)
            total_area = attention_map.shape[0] * attention_map.shape[1]
            
            if area > 0.05 * total_area:
                avg_attention = attention_map[region_mask].mean()
                regions.append({
                    "bbox": (xmin, ymin, xmax, ymax),
                    "attention_score": float(avg_attention),
                    "area": int(area),
                    "id": region_id
                })
        
        # Sort by attention score (lowest first = most under-explored)
        regions.sort(key=lambda r: r['attention_score'])
        
        return regions
    
    def _generate_multi_scale_crops(
        self,
        image: Image.Image,
        region: Dict,
        scales: List[float]
    ) -> List[Dict]:
        """
        Generate crops at multiple scales centered on region
        """
        xmin, ymin, xmax, ymax = region['bbox']
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        
        crops = []
        for scale in scales:
            # Calculate crop size
            w = int((xmax - xmin) * scale)
            h = int((ymax - ymin) * scale)
            
            # Center on region, clip to image bounds
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(image.width, x1 + w)
            y2 = min(image.height, y1 + h)
            
            # Crop image
            cropped = image.crop((x1, y1, x2, y2))
            
            crops.append({
                "image": cropped,
                "bbox": (x1, y1, x2, y2),
                "scale": scale,
                "region_id": region.get("id")
            })
        
        return crops
    
    def _generate_grid_crops(
        self,
        image: Image.Image,
        n: int = 3
    ) -> List[Dict]:
        """
        Fallback: Generate grid-based crops
        """
        crops = []
        grid_size = int(np.ceil(np.sqrt(n)))
        
        cell_w = image.width // grid_size
        cell_h = image.height // grid_size
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = min(x1 + cell_w, image.width)
                y2 = min(y1 + cell_h, image.height)
                
                cropped = image.crop((x1, y1, x2, y2))
                
                crops.append({
                    "image": cropped,
                    "bbox": (x1, y1, x2, y2),
                    "scale": 1.0,
                    "region_id": count
                })
                
                count += 1
        
        return crops


class SelfRefinementEngine:
    """
    Main refinement engine combining uncertainty analysis and visual re-attention
    for iterative response improvement.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        self.device = device
        
        # Load Qwen2-VL model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.uncertainty_analyzer = UncertaintyAnalyzer(
            self.model, self.processor, device
        )
        
        self.visual_reattention = VisualReAttention(
            self.model, self.processor, device
        )
        
        print("Model loaded successfully!")
    
    def refine_response(
        self,
        image: Image.Image,
        question: str,
        max_iterations: int = 3,
        uncertainty_threshold: float = 0.15,
        verbose: bool = True
    ) -> RefinementResult:
        """
        Main refinement pipeline
        
        Args:
            image: Input image
            question: Question to answer
            max_iterations: Maximum refinement rounds
            uncertainty_threshold: Stop if uncertainty below this
            verbose: Print progress
        
        Returns:
            RefinementResult with final response and metrics
        """
        history = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SELF-REFINEMENT PIPELINE")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"Max iterations: {max_iterations}")
            print(f"Uncertainty threshold: {uncertainty_threshold}")
            print(f"{'='*60}\n")
        
        # Round 0: Initial response
        if verbose:
            print("ROUND 0: Generating initial response...")
        
        response_0 = self._generate_response(image, question)
        uncertainty_0 = self.uncertainty_analyzer.compute_uncertainty(
            image, question, response_0
        )
        
        if verbose:
            print(f"Response: {response_0}")
            print(f"Uncertainty: {uncertainty_0.overall:.3f}")
            print(f"  - Token level: {uncertainty_0.token_level:.3f}")
            print(f"  - Attention level: {uncertainty_0.attention_level:.3f}")
            print(f"  - Semantic level: {uncertainty_0.semantic_level:.3f}")
            print(f"  - Claim level: {uncertainty_0.claim_level:.3f}")
            print()
        
        history.append({
            "round": 0,
            "response": response_0,
            "uncertainty": uncertainty_0.overall,
            "type": "initial"
        })
        
        current_response = response_0
        current_uncertainty = uncertainty_0.overall
        
        # Iterative refinement
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"ROUND {iteration}: Refining...")
            
            # Check stopping criterion
            if current_uncertainty < uncertainty_threshold:
                if verbose:
                    print(f"Uncertainty below threshold ({uncertainty_threshold}). Stopping.")
                break
            
            # Generate attention-guided crops
            crops = self.visual_reattention.generate_attention_guided_crops(
                image, question, uncertainty_0, max_crops=2
            )
            
            if verbose:
                print(f"Generated {len(crops)} verification crops")
            
            # Verify on crops
            verifications = []
            for i, crop_info in enumerate(crops):
                verification = self._verify_on_crop(
                    crop_info["image"],
                    question,
                    current_response
                )
                verifications.append(verification)
                
                if verbose:
                    print(f"  Crop {i+1}: {verification['response'][:100]}...")
            
            # Integrate verifications
            refined_response = self._integrate_verifications(
                current_response,
                verifications,
                question
            )
            
            # Recalculate uncertainty
            refined_uncertainty_metrics = self.uncertainty_analyzer.compute_uncertainty(
                image, question, refined_response
            )
            refined_uncertainty = refined_uncertainty_metrics.overall
            
            if verbose:
                print(f"Refined response: {refined_response}")
                print(f"New uncertainty: {refined_uncertainty:.3f}")
                print(f"Improvement: {current_uncertainty - refined_uncertainty:.3f}")
                print()
            
            history.append({
                "round": iteration,
                "response": refined_response,
                "uncertainty": refined_uncertainty,
                "verifications": len(verifications),
                "type": "refinement"
            })
            
            # Check for convergence
            improvement = current_uncertainty - refined_uncertainty
            if improvement < 0.05:  # Less than 5% improvement
                if verbose:
                    print("Minimal improvement. Stopping.")
                break
            
            current_response = refined_response
            current_uncertainty = refined_uncertainty
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"REFINEMENT COMPLETE")
            print(f"Final response: {current_response}")
            print(f"Final uncertainty: {current_uncertainty:.3f}")
            print(f"Total improvement: {uncertainty_0.overall - current_uncertainty:.3f}")
            print(f"Iterations used: {len(history)}")
            print(f"{'='*60}\n")
        
        return RefinementResult(
            final_response=current_response,
            final_uncertainty=current_uncertainty,
            iterations=len(history),
            history=history,
            improvement=uncertainty_0.overall - current_uncertainty
        )
    
    def _generate_response(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 150
    ) -> str:
        """Generate response from VLM using Qwen3-VL format"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for consistency
                    temperature=1.0
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error: Could not generate response"
    
    def _verify_on_crop(
        self,
        crop_image: Image.Image,
        original_question: str,
        current_response: str
    ) -> Dict:
        """
        Verify details on a cropped region
        """
        # Generate verification question
        verification_q = f"Looking at this region closely, {original_question}"
        
        crop_response = self._generate_response(crop_image, verification_q)
        
        # Check consistency with current response
        is_consistent = self._check_consistency(current_response, crop_response)
        
        return {
            "question": verification_q,
            "response": crop_response,
            "is_consistent": is_consistent
        }
    
    def _check_consistency(self, response1: str, response2: str) -> bool:
        """
        Simple consistency check via word overlap
        """
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / len(words1 | words2)
        
        return overlap > 0.5  # 50% overlap threshold
    
    def _integrate_verifications(
        self,
        original_response: str,
        verifications: List[Dict],
        question: str
    ) -> str:
        """
        Integrate verification results into refined response
        
        Strategy: If verifications are consistent and different from original,
        use verification info to refine
        """
        if not verifications:
            return original_response
        
        # Check if verifications agree
        verification_responses = [v['response'] for v in verifications]
        
        # If all verifications are consistent with each other
        all_consistent = all(
            self._check_consistency(verification_responses[0], vr)
            for vr in verification_responses[1:]
        )
        
        if all_consistent and verification_responses:
            # Use first verification as basis
            new_response = verification_responses[0]
            
            # If significantly different from original, update
            if not self._check_consistency(original_response, new_response):
                return new_response
        
        # Otherwise keep original
        return original_response



def main_unique_image():
    """Example usage"""

    image_path = "test_image.png"  # Replace with actual image path
    image = Image.open(image_path)
    
    # Initialize system with Qwen3-VL
    engine = SelfRefinementEngine(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    
    question = "How many cars are in this parking lot? Count carefully."
    
    # MODIFICATION: Seuil plus élevé pour forcer le raffinement
    result = engine.refine_response(
        image=image,
        question=question,
        max_iterations=3,
        uncertainty_threshold=0.15,  # Changé de 0.3 à 0.15
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Final Response: {result.final_response}")
    print(f"Final Uncertainty: {result.final_uncertainty:.3f}")
    print(f"Iterations: {result.iterations}")
    print(f"Improvement: {result.improvement:.3f}")
    print("="*60)



if __name__ == "__main__":
    main_unique_image()

