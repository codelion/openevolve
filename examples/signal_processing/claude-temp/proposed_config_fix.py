"""
PROPOSED FIX for OpenEvolve Configuration Bug

Issue: diff_based_evolution=True and allow_full_rewrites=True are incompatible
but no validation prevents this configuration from being used.

PROBLEM:
- Prompt sampler uses allow_full_rewrites to choose template
- Controller uses diff_based_evolution to choose parser  
- These can create contradictory behavior

SOLUTION:
Add validation to Config class __post_init__ method
"""

# Add this to openevolve/config.py in the Config class __post_init__ method:

def __post_init__(self):
    """Post-initialization validation"""
    
    # Validate evolution settings compatibility
    if self.diff_based_evolution and self.allow_full_rewrites:
        raise ValueError(
            "Configuration Error: diff_based_evolution=True and allow_full_rewrites=True "
            "are incompatible. Use one of these combinations:\n"
            "  - diff_based_evolution=True, allow_full_rewrites=False (diff-based evolution)\n"
            "  - diff_based_evolution=False, allow_full_rewrites=True (rewrite-based evolution)\n"
            "  - diff_based_evolution=False, allow_full_rewrites=False (rewrite with diff template)"
        )
    
    # Other existing validations...


# Alternative: Add a helper method to validate and suggest fixes:

def validate_evolution_settings(self) -> None:
    """Validate evolution configuration and provide helpful error messages"""
    
    if self.diff_based_evolution and self.allow_full_rewrites:
        suggested_configs = [
            "# Option 1: Pure diff-based evolution (recommended for iterative improvements)",
            "diff_based_evolution: true", 
            "allow_full_rewrites: false",
            "",
            "# Option 2: Pure rewrite-based evolution (recommended for major changes)",
            "diff_based_evolution: false",
            "allow_full_rewrites: true"
        ]
        
        raise ValueError(
            f"❌ Configuration Error: Incompatible evolution settings detected!\n\n"
            f"Current settings:\n"
            f"  diff_based_evolution: {self.diff_based_evolution}\n" 
            f"  allow_full_rewrites: {self.allow_full_rewrites}\n\n"
            f"🔧 Suggested fixes:\n" + "\n".join(suggested_configs) + "\n\n"
            f"💡 Explanation:\n"
            f"  - diff_based_evolution=True makes the controller parse responses as diff blocks\n"
            f"  - allow_full_rewrites=True makes the prompt ask for complete code rewrites\n" 
            f"  - These create a contradiction: LLM returns complete code but controller expects diffs\n"
        )
