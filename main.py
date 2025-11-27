
"""
Self-Correcting Vision-Language Models via Uncertainty-Guided Visual Re-Attention

Main implementation of the self-refinement system for reducing hallucinations
in open-source VLMs through iterative uncertainty-guided verification.

Author: Research Implementation
License: MIT
"""

import torch
from utils.selfrefinementengine import SelfRefinementEngine
from PIL import Image

def main():
    """Example usage"""
    
    # Initialize system
    engine = SelfRefinementEngine(
        model_name="llava-hf/llava-1.5-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load test image
    image = Image.new('RGB', (224, 224), color='white')  # Placeholder
    question = "How many objects are in this image?"
    
    # Run refinement
    result = engine.refine_response(
        image=image,
        question=question,
        max_iterations=3,
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
    main()