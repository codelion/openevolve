"""
Prompt sampling for OpenEvolve
"""
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""
    
    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(config.template_dir)
        
        # Initialize the random number generator
        random.seed()
        
        logger.info("Initialized prompt sampler")
    
    def build_prompt(
        self,
        current_program: str,
        parent_program: str,
        program_metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        language: str = "python",
        evolution_round: int = 0,
        allow_full_rewrite: bool = False,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM
        
        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs
            language: Programming language
            evolution_round: Current evolution round
            allow_full_rewrite: Whether to allow a full rewrite
            
        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on whether we want a full rewrite
        template_key = "full_rewrite_user" if allow_full_rewrite else "diff_user"
        user_template = self.template_manager.get_template(template_key)
        system_template = self.config.system_message
        
        # Format metrics
        metrics_str = self._format_metrics(program_metrics)
        
        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )
        
        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, language
        )
        
        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)
        
        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
        )
        
        return {
            "system": system_template,
            "user": user_message,
        }
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for the prompt"""
        return "\n".join([f"- {name}: {value:.4f}" for name, value in metrics.items()])
    
    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """Identify potential areas for improvement"""
        # This method could be expanded to include more sophisticated analysis
        # For now, we'll use a simple approach
        
        improvement_areas = []
        
        # Check program length
        if len(current_program) > 500:
            improvement_areas.append("Consider simplifying the code to improve readability and maintainability")
        
        # Check for performance patterns in previous attempts
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]
            metrics_improved = []
            metrics_regressed = []
            
            for metric, value in metrics.items():
                improved = True
                regressed = True
                
                for attempt in recent_attempts:
                    if attempt["metrics"].get(metric, 0) <= value:
                        regressed = False
                    if attempt["metrics"].get(metric, 0) >= value:
                        improved = False
                
                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)
            
            if metrics_improved:
                improvement_areas.append(
                    f"Metrics showing improvement: {', '.join(metrics_improved)}. "
                    "Consider continuing with similar changes."
                )
            
            if metrics_regressed:
                improvement_areas.append(
                    f"Metrics showing regression: {', '.join(metrics_regressed)}. "
                    "Consider reverting or revising recent changes in these areas."
                )
        
        # If we don't have specific improvements to suggest
        if not improvement_areas:
            improvement_areas.append(
                "Focus on optimizing the code for better performance on the target metrics"
            )
        
        return "\n".join([f"- {area}" for area in improvement_areas])
    
    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")
        
        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)):]
        
        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("changes", "Unknown changes")
            
            # Format performance metrics
            performance_str = ", ".join([
                f"{name}: {value:.4f}" 
                for name, value in program.get("metrics", {}).items()
            ])
            
            # Determine outcome based on comparison with parent
            parent_metrics = program.get("parent_metrics", {})
            outcome = "Mixed results"
            
            if all(program.get("metrics", {}).get(m, 0) >= parent_metrics.get(m, 0) 
                  for m in program.get("metrics", {})):
                outcome = "Improvement in all metrics"
            elif all(program.get("metrics", {}).get(m, 0) <= parent_metrics.get(m, 0) 
                    for m in program.get("metrics", {})):
                outcome = "Regression in all metrics"
            
            previous_attempts_str += previous_attempt_template.format(
                attempt_number=attempt_number,
                changes=changes,
                performance=performance_str,
                outcome=outcome,
            ) + "\n\n"
        
        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[:min(self.config.num_top_programs, len(top_programs))]
        
        for i, program in enumerate(selected_top):
            # Extract a snippet (first 10 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:10])
            if len(program_code.split("\n")) > 10:
                program_snippet += "\n# ... (truncated for brevity)"
            
            # Calculate a composite score
            score = sum(program.get("metrics", {}).values()) / max(1, len(program.get("metrics", {})))
            
            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = [
                    f"Performs well on {name} ({value:.4f})" 
                    for name, value in program.get("metrics", {}).items()
                ]
            
            key_features_str = ", ".join(key_features)
            
            top_programs_str += top_program_template.format(
                program_number=i + 1,
                score=f"{score:.4f}",
                language=language,
                program_snippet=program_snippet,
                key_features=key_features_str,
            ) + "\n\n"
        
        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=top_programs_str.strip(),
        )
    
    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template
        
        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)
        
        return result
