"""
Fusion Module
Combines results from multiple sources (classification, logo, OCR, embedding)
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class ResultFusion:
    """Fuses results from multiple identification methods"""
    
    def __init__(self, weights: Dict[str, float] = None, final_top_k: int = 5):
        """
        Initialize result fusion
        
        Args:
            weights: Weights for each method (classification, logo, ocr, embedding)
            final_top_k: Number of final results to return
        """
        self.weights = weights or {
            'classification': 0.4,
            'logo': 0.2,
            'ocr': 0.1,
            'embedding': 0.3
        }
        self.final_top_k = final_top_k
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def fuse_results(self, 
                    classification_results: List[Dict],
                    logo_results: List[Dict],
                    ocr_results: List[Dict],
                    embedding_results: List[Dict]) -> List[Dict]:
        """
        Fuse results from all methods
        
        Args:
            classification_results: Results from classifier
            logo_results: Results from logo detector
            ocr_results: OCR text results
            embedding_results: Results from embedding search
            
        Returns:
            Fused and ranked results
        """
        # Aggregate scores by vehicle identity
        vehicle_scores = defaultdict(lambda: {
            'scores': defaultdict(float),
            'evidence': defaultdict(list),
            'details': {}
        })
        
        # Process classification results
        for result in classification_results:
            key = self._make_key(result.get('make', ''), result.get('model', ''))
            score = result['confidence'] * self.weights['classification']
            vehicle_scores[key]['scores']['classification'] = score
            vehicle_scores[key]['evidence']['classification'].append({
                'confidence': result['confidence'],
                'source': 'classifier'
            })
            vehicle_scores[key]['details'].update({
                'make': result.get('make', 'Unknown'),
                'model': result.get('model', 'Unknown'),
                'year_range': result.get('year_range', 'Unknown')
            })
        
        # Process logo results
        if logo_results:
            logo_brand = self._extract_logo_brand(logo_results)
            if logo_brand:
                # Boost scores for vehicles matching the logo brand
                for key in vehicle_scores:
                    if logo_brand.lower() in key.lower():
                        max_conf = max([r['confidence'] for r in logo_results])
                        score = max_conf * self.weights['logo']
                        vehicle_scores[key]['scores']['logo'] = score
                        vehicle_scores[key]['evidence']['logo'].append({
                            'brand': logo_brand,
                            'confidence': max_conf,
                            'source': 'logo_detector'
                        })
        
        # Process OCR results
        ocr_info = self._extract_ocr_info(ocr_results)
        if ocr_info['brands'] or ocr_info['models']:
            for key in vehicle_scores:
                match_score = 0.0
                matches = []
                
                # Check brand matches
                for brand in ocr_info['brands']:
                    if brand.lower() in key.lower():
                        match_score += 0.5
                        matches.append(f"Brand: {brand}")
                
                # Check model matches
                for model in ocr_info['models']:
                    if model.lower() in key.lower():
                        match_score += 0.5
                        matches.append(f"Model: {model}")
                
                if match_score > 0:
                    score = min(match_score, 1.0) * self.weights['ocr']
                    vehicle_scores[key]['scores']['ocr'] = score
                    vehicle_scores[key]['evidence']['ocr'].append({
                        'matches': matches,
                        'source': 'ocr'
                    })
        
        # Process embedding results
        for result in embedding_results:
            key = self._make_key(result.get('make', ''), result.get('model', ''))
            score = result.get('similarity_score', 0.0) * self.weights['embedding']
            vehicle_scores[key]['scores']['embedding'] = score
            vehicle_scores[key]['evidence']['embedding'].append({
                'similarity': result.get('similarity_score', 0.0),
                'source': 'embedding_search'
            })
            
            # Update details if not already set
            if 'make' not in vehicle_scores[key]['details']:
                vehicle_scores[key]['details'].update({
                    'make': result.get('make', 'Unknown'),
                    'model': result.get('model', 'Unknown'),
                    'year_range': result.get('year_range', 'Unknown')
                })
        
        # Calculate total scores and rank
        ranked_results = []
        for key, data in vehicle_scores.items():
            total_score = sum(data['scores'].values())
            
            result = {
                'vehicle_id': key,
                'total_score': total_score,
                'component_scores': dict(data['scores']),
                'evidence': dict(data['evidence']),
                **data['details']
            }
            ranked_results.append(result)
        
        # Sort by total score
        ranked_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Return top-k
        return ranked_results[:self.final_top_k]
    
    def _make_key(self, make: str, model: str) -> str:
        """Create a unique key for a vehicle"""
        return f"{make}_{model}".lower().replace(' ', '_')
    
    def _extract_logo_brand(self, logo_results: List[Dict]) -> str:
        """Extract the most confident brand from logo results"""
        if not logo_results:
            return ""
        best = max(logo_results, key=lambda x: x['confidence'])
        return best['brand']
    
    def _extract_ocr_info(self, ocr_results: List[Dict]) -> Dict[str, List[str]]:
        """Extract brand and model information from OCR results"""
        from ..modules.ocr import VehicleOCR
        
        # Create temporary OCR instance to use helper methods
        ocr = VehicleOCR()
        
        brands = []
        models = []
        
        if ocr_results:
            brands = ocr.extract_brand_info(ocr_results)
            models = ocr.extract_model_info(ocr_results)
        
        return {'brands': brands, 'models': models}
    
    def format_final_results(self, fused_results: List[Dict]) -> Dict:
        """
        Format final results for presentation
        
        Args:
            fused_results: Fused results
            
        Returns:
            Formatted result dictionary
        """
        if not fused_results:
            return {
                'best_match': None,
                'confidence': 0.0,
                'alternatives': [],
                'evidence': {}
            }
        
        best = fused_results[0]
        alternatives = fused_results[1:] if len(fused_results) > 1 else []
        
        return {
            'best_match': {
                'make': best.get('make', 'Unknown'),
                'model': best.get('model', 'Unknown'),
                'year_range': best.get('year_range', 'Unknown'),
                'confidence': best['total_score']
            },
            'confidence': best['total_score'],
            'alternatives': [
                {
                    'make': alt.get('make', 'Unknown'),
                    'model': alt.get('model', 'Unknown'),
                    'year_range': alt.get('year_range', 'Unknown'),
                    'confidence': alt['total_score']
                }
                for alt in alternatives
            ],
            'evidence': {
                'classification': best['evidence'].get('classification', []),
                'logo': best['evidence'].get('logo', []),
                'ocr': best['evidence'].get('ocr', []),
                'embedding': best['evidence'].get('embedding', [])
            },
            'component_scores': best['component_scores']
        }
