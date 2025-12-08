"""
Visualisation légère des zones de chaleur (heatmaps) pour illustrer l'algorithme
de self-correction SANS charger le modèle VLM complet.

Cette version utilise des heatmaps simulées et des données synthétiques
pour démontrer le fonctionnement de l'algorithme sur des ordinateurs
sans GPU puissant.

Author: Kassoum Sanogo
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# Configuration de style pour publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


class LightweightAttentionVisualizer:
    """
    Visualiseur léger pour démontrer l'algorithme sans charger le modèle VLM.
    Utilise des heatmaps simulées basées sur la saillance visuelle.
    """
    
    def __init__(self):
        """Initialise le visualiseur léger"""
        pass
    
    def visualize_full_pipeline(
        self,
        image_path: str,
        question: str,
        output_dir: str = "visualizations",
        max_iterations: int = 3,
        uncertainty_threshold: float = 0.15
    ):
        """
        Visualise le pipeline complet avec des données simulées
        
        Args:
            image_path: Chemin vers l'image à analyser
            question: Question posée au modèle
            output_dir: Répertoire de sortie
            max_iterations: Nombre maximum d'itérations
            uncertainty_threshold: Seuil de convergence
        """
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger l'image
        image = Image.open(image_path).convert("RGB")
        
        print(f"\n{'='*70}")
        print(f"SELF-CORRECTION PIPELINE VISUALIZATION (Lightweight Mode)")
        print(f"{'='*70}")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"{'='*70}\n")
        
        # Générer les données simulées pour toutes les itérations
        history = self._simulate_refinement_process(
            image, question, max_iterations, uncertainty_threshold
        )
        
        # Visualiser chaque itération
        for iter_data in history:
            output_path = os.path.join(
                output_dir, f"iteration_{iter_data['iteration']}.png"
            )
            self._visualize_iteration(image, question, iter_data, output_path)
        
        # Créer une visualisation combinée
        self._create_summary_visualization(
            image, question, history,
            os.path.join(output_dir, "summary.png")
        )
        
        # Créer un graphique de convergence
        self._plot_convergence(
            history,
            os.path.join(output_dir, "convergence.png")
        )
        
        print(f"\n{'='*70}")
        print(f"✅ Visualizations saved in: {output_dir}/")
        print(f"{'='*70}\n")
        
        return history
    
    def _simulate_refinement_process(
        self,
        image: Image.Image,
        question: str,
        max_iterations: int,
        uncertainty_threshold: float
    ) -> List[Dict]:
        """
        Simule le processus de raffinement avec des données réalistes
        """
        history = []
        
        # Paramètres de simulation réalistes (basés sur vos résultats expérimentaux)
        # Incertitude décroissante: 0.52 → 0.38 → 0.31 → 0.27
        base_uncertainties = [0.52, 0.38, 0.31, 0.27]
        
        # Simulated typical responses (progressive counting towards 21 cars)
        responses = [
            "There are 12 cars in the parking lot.",  # Iteration 0 (initial underestimation)
            "Upon closer examination of the peripheral areas, I count 17 cars in the parking lot, including several that were partially obscured in the shadows.",  # Iteration 1
            "After re-examining the darker regions and corners more carefully, I count 20 cars total in the parking lot.",  # Iteration 2
            "Final count after thorough examination: 21 cars are visible in the parking lot, including those in less prominent areas."  # Iteration 3
        ]
        
        for iteration in range(min(max_iterations + 1, len(base_uncertainties))):
            # Incertitude globale
            overall_unc = base_uncertainties[iteration]
            
            # Composantes d'incertitude (distribution réaliste)
            token_unc = overall_unc * np.random.uniform(0.9, 1.1)
            attention_unc = overall_unc * np.random.uniform(0.8, 1.0)
            semantic_unc = overall_unc * np.random.uniform(0.85, 1.05)
            claim_unc = overall_unc * np.random.uniform(0.7, 0.95)
            
            # Générer la heatmap d'attention simulée
            heatmap = self._generate_simulated_heatmap(image, iteration)
            
            # Identifier les régions sous-explorées
            if iteration < max_iterations and overall_unc > uncertainty_threshold:
                crops = self._identify_and_generate_crops(image, heatmap, iteration)
            else:
                crops = []
            
            iter_data = {
                'iteration': iteration,
                'response': responses[min(iteration, len(responses) - 1)],
                'uncertainty': {
                    'overall': overall_unc,
                    'token_level': token_unc,
                    'attention_level': attention_unc,
                    'semantic_level': semantic_unc,
                    'claim_level': claim_unc
                },
                'heatmap': heatmap,
                'crops': crops
            }
            
            history.append(iter_data)
            
            # Display information
            print(f"📍 ITERATION {iteration}")
            print(f"  Response: {iter_data['response']}")
            print(f"  Overall uncertainty: {overall_unc:.3f}")
            print(f"  - Token entropy: {token_unc:.3f}")
            print(f"  - Attention dispersion: {attention_unc:.3f}")
            print(f"  - Semantic consistency: {semantic_unc:.3f}")
            print(f"  - Claim confidence: {claim_unc:.3f}")
            print(f"  Generated crops: {len(crops)}\n")
            
            # Check convergence
            if overall_unc < uncertainty_threshold:
                print(f"  ✅ Convergence reached (u={overall_unc:.3f} < {uncertainty_threshold})\n")
                break
        
        return history
    
    def _generate_simulated_heatmap(
        self,
        image: Image.Image,
        iteration: int
    ) -> np.ndarray:
        """
        Génère une heatmap d'attention simulée basée sur la saillance visuelle
        
        La heatmap évolue à chaque itération pour montrer la redistribution
        de l'attention vers les régions initialement sous-explorées.
        """
        # Convertir en niveaux de gris
        gray = np.array(image.convert('L'))
        h, w = gray.shape
        
        # Calculer la saillance multi-échelle pour mieux détecter les objets
        saliency = np.zeros_like(gray, dtype=float)
        
        # Saillance à différentes échelles (détecte objets de différentes tailles)
        for sigma in [1, 2, 4]:
            blurred = gaussian_filter(gray.astype(float), sigma=sigma)
            gradient_x = np.abs(np.gradient(blurred, axis=1))
            gradient_y = np.abs(np.gradient(blurred, axis=0))
            saliency += np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normaliser
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Détecter les régions sombres et claires (zones potentielles d'objets)
        # Normaliser l'intensité
        normalized_gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Créer des masques pour différentes zones
        dark_regions = 1 - normalized_gray  # Zones sombres inversées
        bright_regions = normalized_gray     # Zones claires
        
        # Combiner saillance avec contraste local
        combined_saliency = 0.5 * saliency + 0.25 * dark_regions + 0.25 * bright_regions
        
        # À l'itération 0, simuler un biais d'attention central et vers le haut
        # (biais typique des VLMs: focalisent sur centre-haut de l'image)
        center_y, center_x = int(h * 0.4), w // 2  # Centre légèrement vers le haut
        
        # Créer un masque multi-pics pour simuler attention sur plusieurs zones
        y, x = np.ogrid[:h, :w]
        
        if iteration == 0:
            # Itération 0: Forte concentration au centre-haut avec quelques pics secondaires
            # Pic principal (centre-haut)
            main_peak = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
            
            # Pics secondaires (zones évidentes)
            secondary_peaks = np.zeros_like(main_peak)
            # Ajouter des pics aux positions typiquement visibles
            for py, px in [(h//4, w//4), (h//4, 3*w//4), (h//2, w//2)]:
                peak = np.exp(-((x - px)**2 + (y - py)**2) / (2 * (min(h, w) / 6)**2))
                secondary_peaks = np.maximum(secondary_peaks, peak * 0.6)
            
            # Combiner: 70% pic principal, 30% saillance
            attention_bias = 0.7 * main_peak + 0.3 * secondary_peaks
            heatmap = 0.7 * attention_bias + 0.3 * combined_saliency
            
        elif iteration == 1:
            # Itération 1: Redistribution vers les côtés et coins
            # Réduire le biais central
            main_peak = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 3)**2))
            
            # Ajouter attention aux côtés
            side_attention = np.zeros_like(main_peak)
            for py, px in [(h//3, w//6), (h//3, 5*w//6), (2*h//3, w//4), (2*h//3, 3*w//4)]:
                peak = np.exp(-((x - px)**2 + (y - py)**2) / (2 * (min(h, w) / 7)**2))
                side_attention = np.maximum(side_attention, peak * 0.7)
            
            # Équilibrer: 40% centre, 40% côtés, 20% saillance
            heatmap = 0.4 * main_peak + 0.4 * side_attention + 0.2 * combined_saliency
            
        elif iteration == 2:
            # Itération 2: Exploration des coins et zones périphériques
            corner_attention = np.zeros((h, w))
            
            # Coins et bords
            corner_positions = [
                (h//6, w//6), (h//6, 5*w//6),      # Haut gauche/droite
                (h//2, w//8), (h//2, 7*w//8),      # Milieu gauche/droite
                (5*h//6, w//6), (5*h//6, 5*w//6),  # Bas gauche/droite
                (h//3, w//2), (2*h//3, w//2)       # Milieu
            ]
            
            for py, px in corner_positions:
                peak = np.exp(-((x - px)**2 + (y - py)**2) / (2 * (min(h, w) / 8)**2))
                corner_attention = np.maximum(corner_attention, peak * 0.6)
            
            # 30% coins, 70% saillance distribuée
            heatmap = 0.3 * corner_attention + 0.7 * combined_saliency
            
        else:  # iteration >= 3
            # Itération 3+: Couverture uniforme avec focus sur zones à haute saillance
            # Créer une grille d'attention uniforme
            grid_attention = np.ones((h, w)) * 0.3
            
            # Ajouter des pics sur toute l'image en grille
            for i in range(4):
                for j in range(4):
                    py, px = int(h * (i + 0.5) / 4), int(w * (j + 0.5) / 4)
                    peak = np.exp(-((x - px)**2 + (y - py)**2) / (2 * (min(h, w) / 10)**2))
                    grid_attention = np.maximum(grid_attention, peak * 0.5)
            
            # Principalement basé sur la saillance avec grille uniforme
            heatmap = 0.2 * grid_attention + 0.8 * combined_saliency
        
        # Normaliser
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Lisser pour un rendu plus réaliste
        sigma = max(3, 8 - iteration * 2)  # Moins de lissage = plus de détails
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Renormaliser après lissage
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _identify_and_generate_crops(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        iteration: int
    ) -> List[Dict]:
        """
        Identifie les régions sous-explorées et génère des crops
        """
        # Régions avec attention < 25% du max (plus sensible)
        threshold = 0.25 * heatmap.max()
        underexplored_mask = heatmap < threshold
        
        # Analyse des composantes connectées
        labeled, num_features = ndimage.label(underexplored_mask)
        
        crops = []
        img_width, img_height = image.size
        
        # Extraire les régions les plus prometteuses
        for region_id in range(1, min(num_features + 1, 8)):  # Jusqu'à 8 régions
            mask = labeled == region_id
            
            # Calculer la bounding box
            rows, cols = np.where(mask)
            
            if len(rows) < 50:  # Ignorer les régions trop petites (seuil réduit)
                continue
            
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            
            # Ajouter une marge proportionnelle
            margin_y = int((y_max - y_min) * 0.15)
            margin_x = int((x_max - x_min) * 0.15)
            
            y_min = max(0, y_min - margin_y)
            y_max = min(img_height, y_max + margin_y)
            x_min = max(0, x_min - margin_x)
            x_max = min(img_width, x_max + margin_x)
            
            # Calculer le centre de la région
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            
            # Score d'attention moyen de la région
            attention_score = heatmap[y_min:y_max, x_min:x_max].mean()
            
            # Générer des crops à différentes échelles
            scales = [1.2, 1.5, 2.0] if iteration <= 1 else [1.3, 1.8]
            
            for scale in scales:
                # Calculer la taille du crop
                height = int((y_max - y_min) * scale)
                width = int((x_max - x_min) * scale)
                
                # Assurer une taille minimale pour voir les détails
                height = max(height, img_height // 4)
                width = max(width, img_width // 4)
                
                # Centrer le crop
                crop_y_min = max(0, center_y - height // 2)
                crop_y_max = min(img_height, center_y + height // 2)
                crop_x_min = max(0, center_x - width // 2)
                crop_x_max = min(img_width, center_x + width // 2)
                
                # Ajuster si on dépasse les limites
                if crop_y_max - crop_y_min < height:
                    if crop_y_min == 0:
                        crop_y_max = min(img_height, height)
                    else:
                        crop_y_min = max(0, img_height - height)
                
                if crop_x_max - crop_x_min < width:
                    if crop_x_min == 0:
                        crop_x_max = min(img_width, width)
                    else:
                        crop_x_min = max(0, img_width - width)
                
                # Vérifier que les coordonnées sont valides
                if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min:
                    continue  # Ignorer ce crop s'il est invalide
                
                # S'assurer que les coordonnées respectent les limites de l'image
                crop_x_min = max(0, min(crop_x_min, img_width - 1))
                crop_x_max = max(crop_x_min + 1, min(crop_x_max, img_width))
                crop_y_min = max(0, min(crop_y_min, img_height - 1))
                crop_y_max = max(crop_y_min + 1, min(crop_y_max, img_height))
                
                # Vérifier à nouveau après ajustement
                if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min:
                    continue
                
                # Extraire le crop
                try:
                    crop_img = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                except ValueError:
                    # Si le crop échoue quand même, ignorer
                    continue
                
                crops.append({
                    'image': crop_img,
                    'bbox': (crop_x_min, crop_y_min, crop_x_max, crop_y_max),
                    'scale': scale,
                    'attention_score': float(attention_score),
                    'region_id': region_id
                })
        
        # Trier par score d'attention (plus bas = plus sous-exploré)
        crops.sort(key=lambda c: c['attention_score'])
        
        # Retourner plus de crops pour une meilleure couverture
        return crops[:8]  # Jusqu'à 8 crops
    
    def _visualize_iteration(
        self,
        image: Image.Image,
        question: str,
        iteration_data: Dict,
        output_path: str
    ):
        """
        Visualise une itération avec heatmap et crops
        """
        iteration = iteration_data['iteration']
        uncertainty = iteration_data['uncertainty']
        heatmap = iteration_data['heatmap']
        crops = iteration_data['crops']
        
        # Créer la figure
        n_crops = len(crops)
        n_cols = min(4, n_crops + 2)
        n_rows = max(2, (n_crops + 1) // (n_cols - 1) + 1)
        
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[0.3] + [1]*(n_rows-1))
        
        # Titre
        fig.suptitle(
            f"Iteration {iteration} - Uncertainty: {uncertainty['overall']:.3f}",
            fontsize=14, fontweight='bold', y=0.98
        )
        
        # Zone de texte pour question et réponse
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off')
        
        text_content = (
            f"Question: {question}\n\n"
            f"Response: {iteration_data['response']}\n\n"
            f"Uncertainty Metrics:\n"
            f"  • Token entropy: {uncertainty['token_level']:.3f}\n"
            f"  • Attention dispersion: {uncertainty['attention_level']:.3f}\n"
            f"  • Semantic consistency: {uncertainty['semantic_level']:.3f}\n"
            f"  • Claim confidence: {uncertainty['claim_level']:.3f}"
        )
        
        ax_text.text(0.05, 0.5, text_content,
                    transform=ax_text.transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Original image
        ax_img = fig.add_subplot(gs[1, 0])
        ax_img.imshow(image)
        ax_img.set_title("Original Image", fontweight='bold')
        ax_img.axis('off')
        
        # Attention heatmap with overlay
        ax_heat = fig.add_subplot(gs[1, 1])
        
        # Créer l'overlay
        ax_heat.imshow(image, alpha=0.5)
        im = ax_heat.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='bilinear')
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label='Attention')
        
        # Marquer les régions sous-explorées
        if crops:
            for crop in crops:
                bbox = crop['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
                )
                ax_heat.add_patch(rect)
        
        ax_heat.set_title("Attention Heatmap + Targeted Regions", fontweight='bold')
        ax_heat.axis('off')
        
        # Generated crops
        for i, crop in enumerate(crops):
            row = 1 + ((i + 2) // n_cols)
            col = (i + 2) % n_cols
            
            if row < n_rows:
                ax_crop = fig.add_subplot(gs[row, col])
                ax_crop.imshow(crop['image'])
                ax_crop.set_title(
                    f"Crop {i+1} (Scale: {crop['scale']:.1f}x)\n"
                    f"Attention: {crop['attention_score']:.3f}",
                    fontsize=9
                )
                ax_crop.axis('off')
                
                # Bordure colorée selon l'attention
                for spine in ax_crop.spines.values():
                    spine.set_edgecolor('red' if crop['attention_score'] < 0.3 else 'orange')
                    spine.set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  💾 Visualization saved: {output_path}")
    
    def _create_summary_visualization(
        self,
        image: Image.Image,
        question: str,
        history: List[Dict],
        output_path: str
    ):
        """
        Crée une vue d'ensemble de toutes les itérations
        """
        n_iterations = len(history)
        
        fig = plt.figure(figsize=(16, 4 * n_iterations))
        gs = fig.add_gridspec(n_iterations, 4, width_ratios=[1, 1, 1, 1.2])
        
        fig.suptitle(
            f"Self-Correction Process Overview\nQuestion: {question}",
            fontsize=14, fontweight='bold', y=0.995
        )
        
        for i, iter_data in enumerate(history):
            iteration = iter_data['iteration']
            uncertainty = iter_data['uncertainty']
            heatmap = iter_data['heatmap']
            crops = iter_data['crops']
            
            # Image avec boxes
            ax_img = fig.add_subplot(gs[i, 0])
            img_with_boxes = image.copy()
            
            if crops:
                draw = ImageDraw.Draw(img_with_boxes)
                for crop in crops:
                    bbox = crop['bbox']
                    draw.rectangle(bbox, outline='red', width=3)
            
            ax_img.imshow(img_with_boxes)
            ax_img.set_title(f"Iteration {iteration}", fontweight='bold')
            ax_img.axis('off')
            
            # Heatmap
            ax_heat = fig.add_subplot(gs[i, 1])
            ax_heat.imshow(image, alpha=0.5)
            im = ax_heat.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='bilinear')
            ax_heat.set_title("Attention Heatmap", fontweight='bold')
            ax_heat.axis('off')
            
            # Metrics
            ax_metrics = fig.add_subplot(gs[i, 2])
            ax_metrics.axis('off')
            
            metrics_text = (
                f"Uncertainty: {uncertainty['overall']:.3f}\n\n"
                f"Components:\n"
                f"  • Token: {uncertainty['token_level']:.3f}\n"
                f"  • Attention: {uncertainty['attention_level']:.3f}\n"
                f"  • Semantic: {uncertainty['semantic_level']:.3f}\n"
                f"  • Claims: {uncertainty['claim_level']:.3f}\n\n"
                f"Crops: {len(crops)}"
            )
            
            # Couleur de fond selon convergence
            if uncertainty['overall'] < 0.15:
                bg_color = 'lightgreen'
            elif uncertainty['overall'] < 0.30:
                bg_color = 'lightyellow'
            else:
                bg_color = 'lightcoral'
            
            ax_metrics.text(0.1, 0.5, metrics_text,
                          transform=ax_metrics.transAxes,
                          fontsize=9, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.6))
            
            # Response
            ax_response = fig.add_subplot(gs[i, 3])
            ax_response.axis('off')
            
            response_text = f"Response:\n\n{iter_data['response']}"
            
            ax_response.text(0.05, 0.5, response_text,
                           transform=ax_response.transAxes,
                           fontsize=9, verticalalignment='center',
                           wrap=True,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  💾 Summary saved: {output_path}")
    
    def _plot_convergence(
        self,
        history: List[Dict],
        output_path: str
    ):
        """
        Trace l'évolution de l'incertitude
        """
        iterations = [h['iteration'] for h in history]
        overall = [h['uncertainty']['overall'] for h in history]
        token = [h['uncertainty']['token_level'] for h in history]
        attention = [h['uncertainty']['attention_level'] for h in history]
        semantic = [h['uncertainty']['semantic_level'] for h in history]
        claim = [h['uncertainty']['claim_level'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graph 1: Overall uncertainty
        ax1.plot(iterations, overall, 
                marker='o', markersize=10, linewidth=2.5,
                color='#E74C3C', label='Overall Uncertainty')
        
        ax1.fill_between(iterations, 0, overall,
                        alpha=0.3, color='#E74C3C')
        
        # Annotations
        for i, (x, y) in enumerate(zip(iterations, overall)):
            ax1.annotate(f'{y:.3f}',
                        xy=(x, y), xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Threshold line
        ax1.axhline(y=0.15, color='gray', linestyle='--', 
                   linewidth=2, label='Threshold τ=0.15')
        
        ax1.set_xlabel('Iteration', fontweight='bold')
        ax1.set_ylabel('Uncertainty Score', fontweight='bold')
        ax1.set_title('Overall Uncertainty Convergence', fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.legend()
        ax1.set_ylim([0, max(overall) * 1.1])
        
        # Graph 2: Decomposition
        ax2.plot(iterations, token, marker='s', label='Token entropy', linewidth=2)
        ax2.plot(iterations, attention, marker='^', label='Attention', linewidth=2)
        ax2.plot(iterations, semantic, marker='d', label='Semantic', linewidth=2)
        ax2.plot(iterations, claim, marker='v', label='Claims', linewidth=2)
        
        ax2.set_xlabel('Iteration', fontweight='bold')
        ax2.set_ylabel('Uncertainty Score', fontweight='bold')
        ax2.set_title('Component-wise Decomposition', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  💾 Convergence plot saved: {output_path}")


def main():
    """
    Exemple d'utilisation simple
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualisation légère de l'algorithme (sans GPU)"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the image to analyze'
    )
    parser.add_argument(
        '--question',
        type=str,
        default="How many objects are in this image?",
        help='Question to ask'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations',
        help='Output directory'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum number of iterations'
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # Initialize visualizer
    print("\n🚀 Initializing lightweight visualizer...")
    print("   (No GPU mode - Uses simulated heatmaps)\n")
    
    visualizer = LightweightAttentionVisualizer()
    
    # Lancer la visualisation
    history = visualizer.visualize_full_pipeline(
        image_path=args.image,
        question=args.question,
        output_dir=args.output,
        max_iterations=args.max_iterations,
        uncertainty_threshold=0.15
    )
    
    print("\n✨ Visualization completed successfully!")
    print(f"\n📊 Results:")
    print(f"  - Number of iterations: {len(history)}")
    print(f"  - Initial uncertainty: {history[0]['uncertainty']['overall']:.3f}")
    print(f"  - Final uncertainty: {history[-1]['uncertainty']['overall']:.3f}")
    improvement = history[0]['uncertainty']['overall'] - history[-1]['uncertainty']['overall']
    print(f"  - Improvement: {improvement:.3f} ({improvement/history[0]['uncertainty']['overall']*100:.1f}%)")
    print(f"\n📁 Generated files in '{args.output}/':")
    print(f"  - iteration_*.png (detailed visualizations)")
    print(f"  - summary.png (overview)")
    print(f"  - convergence.png (plots)")


if __name__ == "__main__":
    main()
