

import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from utils.selfrefinementengine import RefinementResult
from utils.uncertaintymetrics import UncertaintyAnalyzer
from utils.visualreattention import VisualReAttention


@dataclass
class RefinementResult:
    """Container for refinement output"""
    final_response: str
    final_uncertainty: float
    iterations: int
    history: List[Dict]
    improvement: float
    
    

class SelfRefinementEngine:
    """
    Main refinement engine combining uncertainty analysis and visual re-attention
    for iterative response improvement.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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
        uncertainty_threshold: float = 0.3,
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
        """Generate response from VLM"""
        try:
            inputs = self.processor(
                text=question,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for consistency
                    temperature=1.0
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part (after question)
            if question in response:
                response = response.split(question)[-1].strip()
            
            return response
            
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


