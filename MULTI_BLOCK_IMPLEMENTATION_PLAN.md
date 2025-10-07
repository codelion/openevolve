# Multi-Block Evolution Support - Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to extend OpenEvolve to support multiple `EVOLVE-BLOCK-START/END` regions within a single file. Currently, OpenEvolve only supports one evolve block per file, but the AlphaEvolve paper suggests support for evolving multiple code sections independently or in coordination.

**Key Insight**: The infrastructure for parsing multiple blocks already exists (`parse_evolve_blocks()` in `code_utils.py`) but is **completely unused** in the current codebase. The system currently passes entire files through the evolution pipeline without any block extraction.

---

## Current Architecture Analysis

### How Evolution Currently Works

1. **Initial Program Loading** ([controller.py:221-224](openevolve/controller.py#L221-L224))
   ```python
   def _load_initial_program(self) -> str:
       """Load the initial program from file"""
       with open(self.initial_program_path, "r") as f:
           return f.read()
   ```
   - Loads entire file as string
   - No block parsing occurs
   - Stores full file in `Program.code`

2. **Prompt Building** ([iteration.py:62-64](openevolve/iteration.py#L62-L64))
   ```python
   prompt = prompt_sampler.build_prompt(
       current_program=parent.code,  # FULL FILE CONTENT
       parent_program=parent.code,
       ...
   )
   ```
   - Sends entire file to LLM in prompt
   - Template shows full code in markdown block
   - LLM sees everything, including non-evolved sections

3. **LLM Response Parsing** ([iteration.py:86-105](openevolve/iteration.py#L86-L105))
   ```python
   if config.diff_based_evolution:
       diff_blocks = extract_diffs(llm_response)
       child_code = apply_diff(parent.code, llm_response)  # Applies to FULL FILE
   else:
       new_code = parse_full_rewrite(llm_response, config.language)
       child_code = new_code  # Replaces FULL FILE
   ```
   - `apply_diff()` works on entire file content
   - No awareness of evolve block boundaries
   - LLM can modify anything in the file

4. **Storage** ([database.py:44-73](openevolve/database.py#L44-L73))
   ```python
   @dataclass
   class Program:
       id: str
       code: str  # FULL FILE AS STRING
       language: str = "python"
       # ... no block metadata
   ```
   - Programs stored as monolithic code strings
   - No metadata about block structure

### The Unused Infrastructure

**File**: [openevolve/utils/code_utils.py:9-37](openevolve/utils/code_utils.py#L9-L37)

```python
def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    # ... WORKING IMPLEMENTATION EXISTS
    # ... BUT NEVER CALLED ANYWHERE
```

**Key Finding**: This function can already extract multiple blocks! It just needs to be integrated into the pipeline.

---

## Why This Is Challenging

### Problem 1: Prompt Context Management

**Current**: LLM sees entire file
```python
# helper.py - current state
import numpy as np

def helper():
    return 42

# EVOLVE-BLOCK-START
def optimize_me():
    return slow_algorithm()
# EVOLVE-BLOCK-END
```

**Multi-block scenario**: What should LLM see?
```python
# multi_block.py
import numpy as np

# EVOLVE-BLOCK-START:preprocessing
def preprocess(data):
    return data.normalize()
# EVOLVE-BLOCK-END

def helper():
    return 42

# EVOLVE-BLOCK-START:model
def train_model(data):
    return fit(data)
# EVOLVE-BLOCK-END
```

**Options**:
1. Show LLM **only the blocks** (loses context about helper functions)
2. Show LLM **full file** with blocks highlighted (current approach, works for multi-block)
3. Show LLM **blocks + surrounding context** (complex to implement)

### Problem 2: Diff Application

**Current**: Diffs use SEARCH/REPLACE on full file
```
<<<<<<< SEARCH
def optimize_me():
    return slow_algorithm()
=======
def optimize_me():
    return fast_algorithm()
>>>>>>> REPLACE
```

**Multi-block challenge**: Which block does this target?
- Could be in block 0 or block 1 or block 2
- Need to track which block(s) the diff modifies
- Need to validate diff doesn't touch non-evolved regions

**Solution Options**:
1. **Explicit block markers** in diff format:
   ```
   <<<<<<< SEARCH [BLOCK:preprocessing]
   def preprocess(data):
       return data.normalize()
   =======
   def preprocess(data):
       return data.standardize()
   >>>>>>> REPLACE
   ```

2. **Implicit detection**: Apply diff to first matching block only

3. **Independent block evolution**: Evolve one block at a time, no ambiguity

### Problem 3: Code Reconstruction

After LLM modifies blocks, need to reassemble file:

```
[Header/imports - unchanged]
{{BLOCK_0_PLACEHOLDER}}
[Middle code - unchanged]
{{BLOCK_1_PLACEHOLDER}}
[Footer - unchanged]
```

Need reliable template system that preserves:
- Indentation
- Line numbers (for error reporting)
- Comments outside blocks
- Import statements

---

## Detailed Implementation Plan

### Phase 1: Core Utilities (Foundation)

#### File: `openevolve/utils/code_utils.py`

**New Function 1: Block Extraction with Template**
```python
@dataclass
class BlockInfo:
    """Metadata about an evolve block"""
    index: int           # 0-based block number
    name: Optional[str]  # User-provided name (if using :name syntax)
    start_line: int      # Line number where block starts
    end_line: int        # Line number where block ends
    content: str         # Actual code inside the block
    indentation: str     # Leading whitespace to preserve

def extract_evolve_blocks(code: str) -> Tuple[str, List[BlockInfo]]:
    """
    Extract evolve blocks and create a template for reconstruction.

    Args:
        code: Full source file

    Returns:
        (template, blocks):
            template: Code with blocks replaced by {{BLOCK_N}} placeholders
            blocks: List of BlockInfo with extracted block content

    Example:
        Input code:
            import os
            # EVOLVE-BLOCK-START
            def foo(): pass
            # EVOLVE-BLOCK-END
            print("done")

        Returns:
            ("import os\n{{BLOCK_0}}\nprint(\"done\")",
             [BlockInfo(index=0, content="def foo(): pass", ...)])
    """
    lines = code.split("\n")
    blocks = []
    template_lines = []

    in_block = False
    current_block_index = 0
    current_block_lines = []
    block_start = -1
    block_name = None

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            # Check for optional name: # EVOLVE-BLOCK-START:name
            if ":" in line:
                block_name = line.split(":", 1)[1].strip()

            in_block = True
            block_start = i
            current_block_lines = []

            # Keep the marker line in template for later reconstruction
            template_lines.append(f"{{{{BLOCK_{current_block_index}}}}}")

        elif "# EVOLVE-BLOCK-END" in line and in_block:
            # Extract indentation from first line of block
            indentation = ""
            if current_block_lines:
                indentation = current_block_lines[0][:len(current_block_lines[0]) - len(current_block_lines[0].lstrip())]

            blocks.append(BlockInfo(
                index=current_block_index,
                name=block_name,
                start_line=block_start,
                end_line=i,
                content="\n".join(current_block_lines),
                indentation=indentation
            ))

            in_block = False
            current_block_index += 1
            block_name = None

        elif in_block:
            current_block_lines.append(line)
        else:
            template_lines.append(line)

    template = "\n".join(template_lines)
    return template, blocks
```

**New Function 2: Code Reconstruction**
```python
def reconstruct_code(template: str, evolved_blocks: List[str],
                     original_blocks: List[BlockInfo]) -> str:
    """
    Reconstruct full code from template and evolved blocks.

    Args:
        template: Code template with {{BLOCK_N}} placeholders
        evolved_blocks: List of evolved block contents (strings)
        original_blocks: Original BlockInfo objects (for indentation, markers)

    Returns:
        Full reconstructed code with markers restored

    Example:
        template = "import os\n{{BLOCK_0}}\nprint('done')"
        evolved_blocks = ["def foo():\n    return 42"]
        original_blocks = [BlockInfo(indentation="", ...)]

        Returns:
            "import os\n# EVOLVE-BLOCK-START\ndef foo():\n    return 42\n# EVOLVE-BLOCK-END\nprint('done')"
    """
    result = template

    for i, (evolved_content, original_info) in enumerate(zip(evolved_blocks, original_blocks)):
        placeholder = f"{{{{BLOCK_{i}}}}}"

        # Reconstruct block with markers
        block_name_suffix = f":{original_info.name}" if original_info.name else ""
        reconstructed_block = f"# EVOLVE-BLOCK-START{block_name_suffix}\n"
        reconstructed_block += evolved_content
        reconstructed_block += "\n# EVOLVE-BLOCK-END"

        # Replace placeholder
        result = result.replace(placeholder, reconstructed_block)

    return result
```

**Modified Function: Block-Aware Diff Application**
```python
def apply_diff_to_blocks(template: str, blocks: List[BlockInfo],
                         diff_text: str) -> Tuple[str, List[str]]:
    """
    Apply diffs to specific blocks, handling block-aware diff format.

    Supports two formats:
    1. Explicit: <<<<<<< SEARCH [BLOCK:name]
    2. Implicit: Searches all blocks for match

    Args:
        template: File template with placeholders
        blocks: List of BlockInfo objects
        diff_text: LLM response with diffs

    Returns:
        (new_template, new_blocks): Updated template and evolved block contents

    Raises:
        ValueError: If diff targets non-evolved code or multiple blocks ambiguously
    """
    # Extract diff blocks (existing function)
    diff_blocks = extract_diffs(diff_text)

    # Check if diffs use explicit block markers
    evolved_block_contents = [block.content for block in blocks]

    for search_text, replace_text in diff_blocks:
        # Check for explicit block marker
        block_target = None
        if "[BLOCK:" in search_text:
            # Extract block name/index from marker
            match = re.search(r'\[BLOCK:([^\]]+)\]', search_text)
            if match:
                block_identifier = match.group(1)
                search_text = search_text.replace(f"[BLOCK:{block_identifier}]", "").strip()

                # Find matching block
                for i, block in enumerate(blocks):
                    if (block.name and block.name == block_identifier) or str(i) == block_identifier:
                        block_target = i
                        break

        # Apply diff to targeted block or search all blocks
        if block_target is not None:
            # Apply to specific block
            block_content = evolved_block_contents[block_target]
            if search_text in block_content:
                evolved_block_contents[block_target] = block_content.replace(search_text, replace_text)
            else:
                raise ValueError(f"Search text not found in block {block_target}")
        else:
            # Search all blocks
            matches = []
            for i, content in enumerate(evolved_block_contents):
                if search_text in content:
                    matches.append(i)

            if len(matches) == 0:
                # Check if diff targets non-evolved code (ERROR)
                if search_text in template:
                    raise ValueError(
                        f"Diff targets non-evolved code! SEARCH text found outside evolve blocks. "
                        f"Only code inside EVOLVE-BLOCK-START/END can be modified."
                    )
                raise ValueError(f"Search text not found in any block")
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous diff: search text found in {len(matches)} blocks. "
                    f"Use [BLOCK:name] marker to specify target block."
                )
            else:
                # Apply to single matching block
                block_idx = matches[0]
                evolved_block_contents[block_idx] = evolved_block_contents[block_idx].replace(
                    search_text, replace_text
                )

    return template, evolved_block_contents
```

**Validation Function**
```python
def validate_block_structure(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that evolve block markers are properly structured.

    Checks:
    - Every START has matching END
    - No nested blocks
    - Blocks are not empty

    Returns:
        (is_valid, error_message)
    """
    lines = code.split("\n")
    block_depth = 0
    block_starts = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            if block_depth > 0:
                return False, f"Nested evolve blocks not allowed (line {i+1})"
            block_depth += 1
            block_starts.append(i)
        elif "# EVOLVE-BLOCK-END" in line:
            if block_depth == 0:
                return False, f"EVOLVE-BLOCK-END without matching START (line {i+1})"
            block_depth -= 1

    if block_depth > 0:
        return False, f"Unclosed EVOLVE-BLOCK-START (line {block_starts[-1]+1})"

    # Check for empty blocks
    template, blocks = extract_evolve_blocks(code)
    for block in blocks:
        if not block.content.strip():
            return False, f"Empty evolve block at line {block.start_line}"

    return True, None
```

---

### Phase 2: Data Model Updates

#### File: `openevolve/database.py`

**Option A: Extend Program dataclass with block metadata**
```python
@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str  # Still stores FULL file (with markers)
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW: Block structure information
    block_count: int = 1  # Number of evolve blocks (1 for backward compat)
    blocks_metadata: Optional[List[Dict[str, Any]]] = None
    # Structure: [
    #   {"index": 0, "name": "preprocessing", "start_line": 5, "end_line": 20},
    #   {"index": 1, "name": "model", "start_line": 30, "end_line": 50}
    # ]

    # Prompts
    prompts: Optional[Dict[str, Any]] = None

    # Artifact storage
    artifacts_json: Optional[str] = None
    artifact_dir: Optional[str] = None

    def get_block_info(self) -> Tuple[str, List[BlockInfo]]:
        """Extract blocks from stored code using parse_evolve_blocks"""
        from openevolve.utils.code_utils import extract_evolve_blocks
        return extract_evolve_blocks(self.code)

    def is_multi_block(self) -> bool:
        """Check if this program has multiple evolve blocks"""
        return self.block_count > 1
```

**Option B: Use existing metadata field (simpler, less invasive)**
```python
# No changes to Program dataclass structure

# Convention: Store block info in metadata
program = Program(
    code="...",
    metadata={
        "blocks": {
            "count": 2,
            "blocks": [
                {"index": 0, "name": "preprocessing", "start_line": 5, "end_line": 20},
                {"index": 1, "name": "model", "start_line": 30, "end_line": 50}
            ]
        }
    }
)
```

**Recommendation**: Use Option A for type safety and explicit API, but make it backward compatible (defaults to single block).

---

### Phase 3: Configuration Updates

#### File: `openevolve/config.py`

```python
@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # ... existing fields ...

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000

    # NEW: Multi-block evolution settings
    multi_block_evolution: bool = False
    """Enable evolution of multiple code blocks within a single file"""

    block_evolution_strategy: str = "all"
    """
    Strategy for evolving multiple blocks:
    - "all": Evolve all blocks together in each iteration (coordinated)
    - "random": Randomly select N blocks to evolve per iteration
    - "sequential": Cycle through blocks one at a time
    - "independent": Treat each block as separate program (creates N evolution streams)
    """

    blocks_per_iteration: int = 1
    """Number of blocks to evolve per iteration (when strategy != 'all')"""

    require_explicit_block_markers: bool = True
    """Require LLM to use [BLOCK:name] markers in diffs for multi-block programs"""

    allow_cross_block_context: bool = True
    """Show LLM the full file context (all blocks + surrounding code) or just selected blocks"""
```

#### File: `openevolve/config.yaml` (example)

```yaml
# Example configuration for multi-block evolution
multi_block_evolution: true
block_evolution_strategy: "all"  # or "random", "sequential", "independent"
blocks_per_iteration: 2
require_explicit_block_markers: true
allow_cross_block_context: true

# Rest of config...
llm:
  models:
    - name: "gpt-4"
      weight: 1.0
```

---

### Phase 4: Prompt Engineering

#### File: `openevolve/prompt/templates.py`

**New Template: Multi-Block Diff Evolution**
```python
MULTI_BLOCK_DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}
- Number of evolvable blocks: {num_blocks}
- Blocks to evolve this iteration: {target_blocks}

{artifacts}

# Program Evolution History
{evolution_history}

# File Structure Overview
```{language}
{file_structure}
```

# Evolvable Code Blocks
{blocks_display}

# Task
This program contains {num_blocks} separate evolvable code blocks marked with EVOLVE-BLOCK-START/END.
You are tasked with improving the following block(s): **{target_blocks}**

Focus your improvements on the specified block(s) to increase performance metrics.

**IMPORTANT RULES**:
1. You MUST only modify code inside the EVOLVE-BLOCK-START/END markers
2. Code outside these blocks (imports, helpers, etc.) cannot be changed
3. Use the exact SEARCH/REPLACE diff format with block markers:

<<<<<<< SEARCH [BLOCK:block_name]
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE [BLOCK:block_name]

**Example** (modifying the "preprocessing" block):
<<<<<<< SEARCH [BLOCK:preprocessing]
def preprocess(data):
    return data.normalize()
=======
def preprocess(data):
    # Improved: standardize instead of normalize
    return (data - data.mean()) / data.std()
>>>>>>> REPLACE [BLOCK:preprocessing]

You can suggest multiple changes across different blocks.
Each SEARCH section must exactly match code in the current program's evolve blocks.

**Block Names**:
{block_names_list}

Be thoughtful about your changes and explain your reasoning thoroughly.
"""

# Helper template for displaying file structure (shows placeholders for blocks)
FILE_STRUCTURE_TEMPLATE = """# Imports and setup
{pre_block_code}

# EVOLVE-BLOCK-START:{block_0_name}
[Block 0: {block_0_name} - {block_0_lines} lines]
# EVOLVE-BLOCK-END

{inter_block_code_0}

# EVOLVE-BLOCK-START:{block_1_name}
[Block 1: {block_1_name} - {block_1_lines} lines]
# EVOLVE-BLOCK-END

{post_block_code}
"""

# Helper template for displaying individual blocks
BLOCK_DISPLAY_TEMPLATE = """## Block {block_index}: {block_name}
**Lines**: {start_line}-{end_line}
**Size**: {line_count} lines
**Status**: {evolution_status}

```{language}
{block_content}
```
"""
```

**Update Existing Templates**
```python
# Modify default diff template to support optional block markers
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

{block_specific_instructions}

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""
```

#### File: `openevolve/prompt/sampler.py`

**Modify `build_prompt()` to support multi-block**
```python
class PromptSampler:
    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        feature_dimensions: Optional[List[str]] = None,

        # NEW: Multi-block parameters
        multi_block_mode: bool = False,
        target_block_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Build a prompt for the LLM (with multi-block support)"""

        # Detect if program has multiple blocks
        from openevolve.utils.code_utils import extract_evolve_blocks, validate_block_structure

        # Validate block structure
        is_valid, error = validate_block_structure(current_program)
        if not is_valid:
            logger.error(f"Invalid block structure: {error}")
            # Fall back to treating as single block
            multi_block_mode = False

        # Extract blocks if in multi-block mode
        if multi_block_mode:
            template, blocks = extract_evolve_blocks(current_program)
            num_blocks = len(blocks)

            if num_blocks <= 1:
                # Only one block, use standard template
                multi_block_mode = False
            else:
                # Select which blocks to evolve this iteration
                if target_block_indices is None:
                    # Default: evolve all blocks
                    target_block_indices = list(range(num_blocks))

                # Build multi-block specific prompt sections
                file_structure = self._build_file_structure_display(template, blocks)
                blocks_display = self._build_blocks_display(blocks, target_block_indices, language)
                block_names_list = ", ".join(
                    [f"{i}:{blocks[i].name or f'block_{i}'}" for i in target_block_indices]
                )
                target_blocks_str = ", ".join(
                    [blocks[i].name or f"Block {i}" for i in target_block_indices]
                )

        # Select appropriate template
        if multi_block_mode:
            user_template_key = "multi_block_diff_user"
        elif template_key:
            user_template_key = template_key
        elif self.user_template_override:
            user_template_key = self.user_template_override
        else:
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # ... rest of existing prompt building logic ...

        # Format the final user message
        format_args = {
            "metrics": metrics_str,
            "fitness_score": f"{fitness_score:.4f}",
            "feature_coords": feature_coords,
            "feature_dimensions": ", ".join(feature_dimensions) if feature_dimensions else "None",
            "improvement_areas": improvement_areas,
            "evolution_history": evolution_history,
            "current_program": current_program,
            "language": language,
            "artifacts": artifacts_section,
        }

        # Add multi-block specific fields
        if multi_block_mode:
            format_args.update({
                "num_blocks": num_blocks,
                "target_blocks": target_blocks_str,
                "file_structure": file_structure,
                "blocks_display": blocks_display,
                "block_names_list": block_names_list,
            })
        else:
            format_args["block_specific_instructions"] = ""

        user_message = user_template.format(**format_args, **kwargs)

        return {
            "system": system_message,
            "user": user_message,
        }

    def _build_file_structure_display(self, template: str, blocks: List[BlockInfo]) -> str:
        """Create a high-level view of file structure with block placeholders"""
        # Show template with block summaries instead of full content
        result = template
        for block in blocks:
            placeholder = f"{{{{BLOCK_{block.index}}}}}"
            summary = f"[Block {block.index}: {block.name or 'unnamed'} - {len(block.content.split(chr(10)))} lines]"
            result = result.replace(placeholder, summary)
        return result

    def _build_blocks_display(self, blocks: List[BlockInfo],
                              target_indices: List[int], language: str) -> str:
        """Build detailed display of target blocks"""
        block_template = self.template_manager.get_template("block_display")

        displays = []
        for idx in target_indices:
            block = blocks[idx]
            display = block_template.format(
                block_index=idx,
                block_name=block.name or f"block_{idx}",
                start_line=block.start_line,
                end_line=block.end_line,
                line_count=len(block.content.split("\n")),
                evolution_status="TARGET (will be evolved)",
                language=language,
                block_content=block.content,
            )
            displays.append(display)

        return "\n\n".join(displays)
```

---

### Phase 5: Iteration Logic Updates

#### File: `openevolve/iteration.py`

**Modify `run_iteration_with_shared_db()`**
```python
async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    """Run a single iteration using shared memory database"""
    logger = logging.getLogger(__name__)

    try:
        # Sample parent and inspirations from database
        parent, inspirations = database.sample(num_inspirations=config.prompt.num_top_programs)

        # Get artifacts for the parent program if available
        parent_artifacts = database.get_artifacts(parent.id)

        # NEW: Check if parent has multiple blocks
        from openevolve.utils.code_utils import extract_evolve_blocks, validate_block_structure

        multi_block_mode = False
        template = None
        blocks = None
        target_block_indices = None

        if config.multi_block_evolution:
            is_valid, error = validate_block_structure(parent.code)
            if is_valid:
                template, blocks = extract_evolve_blocks(parent.code)
                if len(blocks) > 1:
                    multi_block_mode = True

                    # Determine which blocks to evolve this iteration
                    target_block_indices = _select_blocks_for_iteration(
                        blocks,
                        config.block_evolution_strategy,
                        config.blocks_per_iteration,
                        iteration
                    )

                    logger.info(
                        f"Iteration {iteration+1}: Multi-block mode active. "
                        f"Evolving blocks: {[blocks[i].name or i for i in target_block_indices]}"
                    )

        # Get island-specific top programs for prompt context
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

        # Build prompt (now with multi-block awareness)
        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,

            # NEW: Multi-block parameters
            multi_block_mode=multi_block_mode,
            target_block_indices=target_block_indices,
        )

        result = Result(parent=parent)
        iteration_start = time.time()

        # Generate code modification
        llm_response = await llm_ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )

        # Parse the response (now with multi-block support)
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response)

            if not diff_blocks:
                logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                return None

            # NEW: Apply diffs with block awareness
            if multi_block_mode:
                from openevolve.utils.code_utils import apply_diff_to_blocks, reconstruct_code

                try:
                    # Apply diffs to specific blocks
                    _, evolved_block_contents = apply_diff_to_blocks(template, blocks, llm_response)

                    # Reconstruct full file
                    child_code = reconstruct_code(template, evolved_block_contents, blocks)

                    changes_summary = format_diff_summary(diff_blocks)
                    if config.require_explicit_block_markers:
                        # Validate that diffs used block markers
                        has_markers = any("[BLOCK:" in search for search, _ in diff_blocks)
                        if not has_markers and len(blocks) > 1:
                            logger.warning(
                                f"Iteration {iteration+1}: Multi-block program but diffs "
                                f"lack [BLOCK:name] markers. This may cause ambiguity."
                            )

                except ValueError as e:
                    logger.error(f"Iteration {iteration+1}: Block diff application failed: {e}")
                    return None
            else:
                # Standard single-block diff application
                child_code = apply_diff(parent.code, llm_response)
                changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite
            new_code = parse_full_rewrite(llm_response, config.language)

            if not new_code:
                logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                return None

            # NEW: Validate block structure is maintained in rewrite
            if multi_block_mode:
                is_valid, error = validate_block_structure(new_code)
                if not is_valid:
                    logger.warning(
                        f"Iteration {iteration+1}: Rewritten code has invalid block structure: {error}"
                    )
                    return None

                # Check that number of blocks is preserved
                _, new_blocks = extract_evolve_blocks(new_code)
                if len(new_blocks) != len(blocks):
                    logger.warning(
                        f"Iteration {iteration+1}: Rewrite changed number of blocks "
                        f"({len(blocks)} -> {len(new_blocks)}). Rejecting."
                    )
                    return None

            child_code = new_code
            changes_summary = "Full rewrite"

        # ... rest of existing evaluation and program creation logic ...

        # Create a child program (with block metadata)
        result.child_program = Program(
            id=child_id,
            code=child_code,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,

            # NEW: Preserve block metadata
            block_count=len(blocks) if multi_block_mode else 1,
            blocks_metadata=[
                {
                    "index": block.index,
                    "name": block.name,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                }
                for block in blocks
            ] if multi_block_mode else None,

            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "evolved_blocks": target_block_indices if multi_block_mode else None,
            },
            prompts={
                template_key: {
                    "system": prompt["system"],
                    "user": prompt["user"],
                    "responses": [llm_response] if llm_response is not None else [],
                }
            } if database.config.log_prompts else None,
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        return None


def _select_blocks_for_iteration(
    blocks: List[BlockInfo],
    strategy: str,
    blocks_per_iteration: int,
    iteration: int
) -> List[int]:
    """
    Select which blocks to evolve in this iteration based on strategy.

    Args:
        blocks: List of available blocks
        strategy: Selection strategy ("all", "random", "sequential", "independent")
        blocks_per_iteration: How many blocks to select
        iteration: Current iteration number

    Returns:
        List of block indices to evolve
    """
    num_blocks = len(blocks)

    if strategy == "all":
        # Evolve all blocks together
        return list(range(num_blocks))

    elif strategy == "random":
        # Randomly select N blocks
        import random
        n = min(blocks_per_iteration, num_blocks)
        return sorted(random.sample(range(num_blocks), n))

    elif strategy == "sequential":
        # Cycle through blocks one at a time
        block_idx = iteration % num_blocks
        return [block_idx]

    elif strategy == "independent":
        # Each iteration evolves one block independently
        # This could be extended to maintain separate populations per block
        block_idx = iteration % num_blocks
        return [block_idx]

    else:
        logger.warning(f"Unknown block selection strategy: {strategy}. Using 'all'.")
        return list(range(num_blocks))
```

#### File: `openevolve/process_parallel.py`

Similar changes needed to the parallel worker version (simplified for brevity):

```python
# In _worker_iteration_impl():

# Add block detection and selection logic
if _worker_config.multi_block_evolution:
    # ... same logic as iteration.py ...
    pass

# Update prompt building with multi-block params
prompt = _worker_prompt_sampler.build_prompt(
    # ... existing params ...
    multi_block_mode=multi_block_mode,
    target_block_indices=target_block_indices,
)

# Update diff application with block awareness
if multi_block_mode:
    _, evolved_block_contents = apply_diff_to_blocks(template, blocks, llm_response)
    child_code = reconstruct_code(template, evolved_block_contents, blocks)
else:
    child_code = apply_diff(parent.code, llm_response)
```

---

### Phase 6: Controller Integration

#### File: `openevolve/controller.py`

**Modify `_load_initial_program()` to detect and validate blocks**
```python
def _load_initial_program(self) -> str:
    """Load the initial program from file"""
    with open(self.initial_program_path, "r") as f:
        code = f.read()

    # NEW: Validate and log block structure
    from openevolve.utils.code_utils import extract_evolve_blocks, validate_block_structure

    is_valid, error = validate_block_structure(code)
    if not is_valid:
        logger.error(f"Initial program has invalid block structure: {error}")
        raise ValueError(f"Invalid EVOLVE-BLOCK structure: {error}")

    template, blocks = extract_evolve_blocks(code)
    num_blocks = len(blocks)

    if num_blocks == 0:
        logger.warning(
            "No EVOLVE-BLOCK-START/END markers found in initial program. "
            "The entire file will be treated as evolvable."
        )
        # Could optionally auto-wrap entire file in markers
    elif num_blocks == 1:
        logger.info(f"Initial program has 1 evolve block")
    else:
        logger.info(f"Initial program has {num_blocks} evolve blocks:")
        for block in blocks:
            logger.info(
                f"  - Block {block.index} ({block.name or 'unnamed'}): "
                f"lines {block.start_line}-{block.end_line} "
                f"({len(block.content.split(chr(10)))} lines of code)"
            )

        if not self.config.multi_block_evolution:
            logger.warning(
                f"Found {num_blocks} evolve blocks but multi_block_evolution=False. "
                f"Set multi_block_evolution=True in config to enable multi-block evolution. "
                f"Currently, the entire file will be evolved as one unit."
            )

    return code
```

**Update initial program storage**
```python
async def run(self, iterations: Optional[int] = None, checkpoint_dir: Optional[str] = None):
    # ... existing code ...

    if should_add_initial:
        logger.info("Adding initial program to database")
        initial_program_id = str(uuid.uuid4())

        # Evaluate the initial program
        initial_metrics = await self.evaluator.evaluate_program(
            self.initial_program_code, initial_program_id
        )

        # NEW: Extract block metadata
        from openevolve.utils.code_utils import extract_evolve_blocks
        _, blocks = extract_evolve_blocks(self.initial_program_code)

        initial_program = Program(
            id=initial_program_id,
            code=self.initial_program_code,
            language=self.config.language,
            metrics=initial_metrics,
            iteration_found=start_iteration,

            # NEW: Store block information
            block_count=len(blocks),
            blocks_metadata=[
                {
                    "index": block.index,
                    "name": block.name,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                }
                for block in blocks
            ] if len(blocks) > 0 else None,
        )

        self.database.add(initial_program)
```

---

### Phase 7: Testing Strategy

#### Test Files to Create

**1. `tests/test_multi_block_utils.py`** - Unit tests for block utilities
```python
import unittest
from openevolve.utils.code_utils import (
    extract_evolve_blocks,
    reconstruct_code,
    apply_diff_to_blocks,
    validate_block_structure,
)

class TestMultiBlockUtils(unittest.TestCase):
    def test_extract_single_block(self):
        """Test extracting a single block"""
        code = """
import os

# EVOLVE-BLOCK-START
def foo():
    return 42
# EVOLVE-BLOCK-END

print("done")
"""
        template, blocks = extract_evolve_blocks(code)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].index, 0)
        self.assertIn("def foo():", blocks[0].content)
        self.assertIn("{{BLOCK_0}}", template)
        self.assertIn("import os", template)
        self.assertIn("print(\"done\")", template)

    def test_extract_multiple_blocks(self):
        """Test extracting multiple blocks"""
        code = """
import numpy as np

# EVOLVE-BLOCK-START:preprocessing
def preprocess(data):
    return data.normalize()
# EVOLVE-BLOCK-END

def helper():
    return 42

# EVOLVE-BLOCK-START:model
def train(data):
    return fit(data)
# EVOLVE-BLOCK-END
"""
        template, blocks = extract_evolve_blocks(code)

        self.assertEqual(len(blocks), 2)

        # Check first block
        self.assertEqual(blocks[0].index, 0)
        self.assertEqual(blocks[0].name, "preprocessing")
        self.assertIn("def preprocess", blocks[0].content)

        # Check second block
        self.assertEqual(blocks[1].index, 1)
        self.assertEqual(blocks[1].name, "model")
        self.assertIn("def train", blocks[1].content)

        # Check template
        self.assertIn("{{BLOCK_0}}", template)
        self.assertIn("{{BLOCK_1}}", template)
        self.assertIn("def helper():", template)
        self.assertNotIn("def preprocess", template)
        self.assertNotIn("def train", template)

    def test_reconstruct_code(self):
        """Test reconstructing code from template and blocks"""
        code = """
# EVOLVE-BLOCK-START
def foo():
    return 1
# EVOLVE-BLOCK-END
"""
        template, blocks = extract_evolve_blocks(code)

        # Evolve the block
        evolved = ["def foo():\n    return 42"]

        # Reconstruct
        result = reconstruct_code(template, evolved, blocks)

        self.assertIn("# EVOLVE-BLOCK-START", result)
        self.assertIn("# EVOLVE-BLOCK-END", result)
        self.assertIn("return 42", result)
        self.assertNotIn("return 1", result)

    def test_apply_diff_to_specific_block(self):
        """Test applying diff to a specific block using markers"""
        code = """
# EVOLVE-BLOCK-START:block_a
def a():
    return 1
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START:block_b
def b():
    return 1
# EVOLVE-BLOCK-END
"""
        template, blocks = extract_evolve_blocks(code)

        diff = """
<<<<<<< SEARCH [BLOCK:block_b]
def b():
    return 1
=======
def b():
    return 2
>>>>>>> REPLACE
"""

        _, evolved_blocks = apply_diff_to_blocks(template, blocks, diff)

        # Block A unchanged
        self.assertIn("return 1", evolved_blocks[0])

        # Block B changed
        self.assertIn("return 2", evolved_blocks[1])
        self.assertNotIn("return 1", evolved_blocks[1])

    def test_validate_block_structure_valid(self):
        """Test validation of valid block structure"""
        code = """
# EVOLVE-BLOCK-START
def foo():
    pass
# EVOLVE-BLOCK-END
"""
        is_valid, error = validate_block_structure(code)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_block_structure_nested(self):
        """Test validation catches nested blocks"""
        code = """
# EVOLVE-BLOCK-START
def foo():
    # EVOLVE-BLOCK-START
    pass
    # EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
"""
        is_valid, error = validate_block_structure(code)
        self.assertFalse(is_valid)
        self.assertIn("Nested", error)

    def test_validate_block_structure_unclosed(self):
        """Test validation catches unclosed blocks"""
        code = """
# EVOLVE-BLOCK-START
def foo():
    pass
"""
        is_valid, error = validate_block_structure(code)
        self.assertFalse(is_valid)
        self.assertIn("Unclosed", error)

    def test_apply_diff_rejects_non_evolved_code(self):
        """Test that diffs targeting non-evolved code are rejected"""
        code = """
def helper():
    return 42

# EVOLVE-BLOCK-START
def foo():
    return 1
# EVOLVE-BLOCK-END
"""
        template, blocks = extract_evolve_blocks(code)

        # Try to modify helper function (not in evolve block)
        diff = """
<<<<<<< SEARCH
def helper():
    return 42
=======
def helper():
    return 99
>>>>>>> REPLACE
"""

        with self.assertRaises(ValueError) as ctx:
            apply_diff_to_blocks(template, blocks, diff)

        self.assertIn("non-evolved code", str(ctx.exception).lower())
```

**2. `tests/test_multi_block_integration.py`** - Integration tests
```python
import unittest
import tempfile
import os
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase

class TestMultiBlockIntegration(unittest.TestCase):
    def test_multi_block_program_storage(self):
        """Test that multi-block programs store metadata correctly"""
        code = """
# EVOLVE-BLOCK-START:preprocessing
def preprocess(x):
    return x * 2
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START:model
def train(x):
    return x + 1
# EVOLVE-BLOCK-END
"""
        program = Program(
            id="test-123",
            code=code,
            block_count=2,
            blocks_metadata=[
                {"index": 0, "name": "preprocessing", "start_line": 1, "end_line": 4},
                {"index": 1, "name": "model", "start_line": 6, "end_line": 9},
            ]
        )

        self.assertTrue(program.is_multi_block())
        self.assertEqual(program.block_count, 2)

        # Test serialization
        as_dict = program.to_dict()
        self.assertEqual(as_dict["block_count"], 2)

        # Test deserialization
        restored = Program.from_dict(as_dict)
        self.assertEqual(restored.block_count, 2)
        self.assertEqual(len(restored.blocks_metadata), 2)
```

**3. Example: `examples/multi_block_optimization/`**

Create a working example with multiple blocks:

```python
# examples/multi_block_optimization/initial_program.py
import numpy as np

# EVOLVE-BLOCK-START:data_preprocessing
def preprocess_data(data):
    """Preprocess input data"""
    # Simple normalization
    return (data - data.min()) / (data.max() - data.min())
# EVOLVE-BLOCK-END

def load_data():
    """Helper function - not evolved"""
    return np.random.rand(100, 10)

# EVOLVE-BLOCK-START:model_training
def train_model(data):
    """Train a simple model"""
    # Simple averaging model
    weights = np.mean(data, axis=0)
    return weights
# EVOLVE-BLOCK-END

def evaluate_model(weights, test_data):
    """Helper function - not evolved"""
    predictions = test_data @ weights
    return np.mean(predictions)

# EVOLVE-BLOCK-START:prediction
def predict(model_weights, input_data):
    """Make predictions"""
    # Simple dot product
    return input_data @ model_weights
# EVOLVE-BLOCK-END
```

```python
# examples/multi_block_optimization/evaluator.py
def evaluate(program_path: str) -> dict:
    """Evaluate the multi-block program"""
    import numpy as np
    import importlib.util

    # Load the program
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Generate test data
    train_data = np.random.rand(100, 10)
    test_data = np.random.rand(20, 10)

    try:
        # Test the pipeline
        preprocessed = program.preprocess_data(train_data)
        model = program.train_model(preprocessed)
        predictions = program.predict(model, test_data)

        # Calculate metrics
        preprocessing_quality = 1.0 - np.std(preprocessed)  # Lower std = better normalization
        model_quality = np.mean(np.abs(model))  # Just a simple metric
        prediction_quality = 1.0 / (1.0 + np.std(predictions))  # Lower variance = better

        combined_score = (preprocessing_quality + model_quality + prediction_quality) / 3

        return {
            "combined_score": combined_score,
            "preprocessing_quality": preprocessing_quality,
            "model_quality": model_quality,
            "prediction_quality": prediction_quality,
        }
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": str(e),
        }
```

```yaml
# examples/multi_block_optimization/config.yaml
# Multi-block evolution example configuration

# Enable multi-block evolution
multi_block_evolution: true
block_evolution_strategy: "all"  # Evolve all blocks together
blocks_per_iteration: 3
require_explicit_block_markers: true
allow_cross_block_context: true

# Standard configuration
max_iterations: 100
checkpoint_interval: 10
random_seed: 42
language: python
diff_based_evolution: true

llm:
  api_base: "https://api.openai.com/v1"
  models:
    - name: "gpt-4"
      weight: 1.0
  temperature: 0.7
  max_tokens: 4096

prompt:
  system_message: |
    You are an expert Python programmer tasked with optimizing a data processing pipeline.
    The pipeline has three separate components: data preprocessing, model training, and prediction.
    Each component is marked with EVOLVE-BLOCK markers and can be improved independently.

    Focus on improving numerical stability, efficiency, and accuracy.

database:
  num_islands: 3
  archive_size: 100
  feature_dimensions:
    - "preprocessing_quality"
    - "model_quality"
    - "prediction_quality"
```

---

## Design Decisions & Trade-offs

### Decision 1: Block Naming

**Option A: Auto-generated names** (`block_0`, `block_1`)
- ✅ Simple to implement
- ✅ No syntax changes needed
- ❌ Less meaningful in prompts/logs

**Option B: User-defined names** (`# EVOLVE-BLOCK-START:preprocessing`)
- ✅ More meaningful and maintainable
- ✅ Better for explicit block targeting in diffs
- ❌ Requires syntax extension
- ❌ Need validation for name uniqueness

**Recommendation**: Support both. Optional names with fallback to indices.

### Decision 2: Evolution Scope

**Option A: All blocks together (coordinated)**
- ✅ Preserves block interactions
- ✅ Simpler implementation
- ❌ Higher complexity per iteration
- ❌ Harder for LLM to focus

**Option B: One block at a time (sequential)**
- ✅ Focused evolution per iteration
- ✅ Better LLM performance on smaller changes
- ❌ May miss inter-block optimizations
- ❌ Slower to improve all blocks

**Option C: Configurable strategy**
- ✅ Maximum flexibility
- ✅ Can experiment with different approaches
- ❌ More complex implementation
- ❌ More config surface area

**Recommendation**: Option C - add `block_evolution_strategy` config with multiple strategies.

### Decision 3: Diff Format

**Option A: Explicit block markers required**
```
<<<<<<< SEARCH [BLOCK:preprocessing]
...
>>>>>>> REPLACE [BLOCK:preprocessing]
```
- ✅ Unambiguous
- ✅ Prevents accidental cross-block edits
- ❌ Requires LLM to learn new format
- ❌ More verbose

**Option B: Implicit detection (search all blocks)**
```
<<<<<<< SEARCH
...
>>>>>>> REPLACE
```
- ✅ Backward compatible
- ✅ LLM already knows format
- ❌ Ambiguous if same code in multiple blocks
- ❌ Error-prone

**Option C: Hybrid (markers optional but recommended)**
- ✅ Backward compatible
- ✅ Supports explicit when needed
- ❌ More complex parsing logic

**Recommendation**: Option C - support both, make explicit markers configurable requirement.

### Decision 4: Backward Compatibility

**Critical**: Single-block files must work unchanged.

**Strategy**:
- Default `multi_block_evolution: false`
- When false, ignore block extraction (treat file as monolithic)
- When true, handle both single-block and multi-block files
- All new fields have sensible defaults (e.g., `block_count: 1`)

### Decision 5: Block Dependencies

**Question**: Should system track which blocks depend on each other?

**Option A: No dependency tracking** (Phase 1)
- ✅ Simpler implementation
- ❌ May break inter-block contracts

**Option B: Explicit dependency declarations** (Future enhancement)
```python
# EVOLVE-BLOCK-START:model [DEPENDS:preprocessing]
def train(data):
    # Expects preprocessed data format
```
- ✅ Prevents breaking changes
- ✅ Could inform evolution strategy
- ❌ Complex to implement
- ❌ Requires specification language

**Recommendation**: Start with Option A, add dependency tracking in Phase 2.

---

## Migration Path

### For Existing Users

1. **No breaking changes**: Default config has `multi_block_evolution: false`
2. **Existing single-block examples continue to work unchanged**
3. **Opt-in**: Set `multi_block_evolution: true` to enable new feature
4. **Gradual adoption**: Can add second block to existing example, enable flag

### For New Users

1. **Examples show both modes**: Single-block and multi-block examples
2. **Documentation explains trade-offs**: When to use multi-block vs single
3. **Error messages guide users**: If multiple blocks detected but flag not set, suggest enabling it

---

## Complexity Estimate

### Lines of Code
- **New code**: ~600-800 lines
  - `code_utils.py`: ~250 lines (block extraction, reconstruction, validation)
  - `prompt/templates.py`: ~100 lines (new templates)
  - `prompt/sampler.py`: ~150 lines (multi-block prompt building)
  - `iteration.py`: ~100 lines (block selection, diff application)
  - `config.py`: ~50 lines (new config fields)
  - `database.py`: ~50 lines (Program dataclass updates)
  - `controller.py`: ~50 lines (initial program validation)

- **Modified code**: ~200-300 lines
  - Updates to existing functions
  - Additional parameters

- **Test code**: ~400-500 lines
  - Unit tests: ~250 lines
  - Integration tests: ~150 lines

**Total**: ~1200-1600 lines

### Files Affected
- **Core files modified**: 8
  - `openevolve/utils/code_utils.py`
  - `openevolve/database.py`
  - `openevolve/config.py`
  - `openevolve/prompt/templates.py`
  - `openevolve/prompt/sampler.py`
  - `openevolve/iteration.py`
  - `openevolve/process_parallel.py`
  - `openevolve/controller.py`

- **Test files created**: 2-3
- **Example directories created**: 1-2
- **Documentation files updated**: 3-4
  - `CLAUDE.md`
  - `examples/README.md`
  - Main `README.md`
  - New: `MULTI_BLOCK.md` (feature documentation)

### Time Estimate
- **Experienced developer** (familiar with codebase): 3-4 days
  - Day 1: Core utilities (extract, reconstruct, validate)
  - Day 2: Integration (prompt, iteration, config)
  - Day 3: Testing and debugging
  - Day 4: Documentation and examples

- **Learning the codebase**: +2-3 days
  - Understanding evolution pipeline
  - Understanding prompt system
  - Understanding database/program model

**Total for prep**: 5-7 days to have working implementation

---

## Implementation Phases (Recommended Order)

### Phase 1: Foundation (Day 1)
**Goal**: Get block extraction/reconstruction working in isolation

1. Implement `extract_evolve_blocks()` with tests
2. Implement `reconstruct_code()` with tests
3. Implement `validate_block_structure()` with tests
4. Add `BlockInfo` dataclass
5. Write comprehensive unit tests

**Success Criteria**: Can extract blocks from multi-block file and reconstruct perfectly

### Phase 2: Data Model (Day 2 morning)
**Goal**: Programs can store block metadata

1. Add `block_count` and `blocks_metadata` to `Program`
2. Update `Program.to_dict()` and `from_dict()`
3. Add config fields to `Config`
4. Test serialization/deserialization

**Success Criteria**: Multi-block programs serialize/deserialize correctly

### Phase 3: Prompts (Day 2 afternoon)
**Goal**: LLM sees multi-block structure in prompts

1. Create `MULTI_BLOCK_DIFF_USER_TEMPLATE`
2. Add `_build_file_structure_display()` to PromptSampler
3. Add `_build_blocks_display()` to PromptSampler
4. Update `build_prompt()` with multi-block params
5. Test prompt generation manually

**Success Criteria**: Multi-block prompt renders correctly with block structure

### Phase 4: Evolution Logic (Day 3)
**Goal**: Diffs apply to specific blocks

1. Implement `apply_diff_to_blocks()`
2. Implement `_select_blocks_for_iteration()`
3. Update `run_iteration_with_shared_db()` with block logic
4. Update parallel worker version
5. Test with mock LLM responses

**Success Criteria**: Can apply block-aware diffs and reconstruct correctly

### Phase 5: Controller Integration (Day 4 morning)
**Goal**: Initial program loading validates blocks

1. Update `_load_initial_program()` to detect blocks
2. Add block validation on startup
3. Update initial program storage with metadata
4. Add logging for block information

**Success Criteria**: System detects and logs multi-block structure on startup

### Phase 6: Testing & Examples (Day 4 afternoon - Day 5)
**Goal**: Prove it works end-to-end

1. Create `examples/multi_block_optimization/`
2. Write integration tests
3. Run full evolution on example
4. Debug and fix issues

**Success Criteria**: Example evolves successfully with multiple blocks

### Phase 7: Documentation (Day 5+)
**Goal**: Users understand how to use feature

1. Update `CLAUDE.md` with multi-block section
2. Update `examples/README.md` with best practices
3. Create `MULTI_BLOCK.md` feature guide
4. Add docstrings to all new functions

**Success Criteria**: Clear documentation with examples

---

## Presentation Strategy

### What to Prepare

1. **Architecture diagram** showing data flow:
   ```
   Initial Program
        ↓
   [Block Extraction] → Template + Blocks
        ↓
   [Prompt Building] → Multi-block aware prompt
        ↓
   [LLM Generation] → Block-targeted diffs
        ↓
   [Diff Application] → Evolved blocks
        ↓
   [Reconstruction] → Complete program
        ↓
   [Evaluation] → Metrics
   ```

2. **Code samples** to walk through:
   - Show `extract_evolve_blocks()` implementation
   - Show template example
   - Show diff with `[BLOCK:name]` marker
   - Show reconstructed result

3. **Design rationale** document:
   - Why blocks instead of separate files?
   - Why template-based reconstruction?
   - Why explicit block markers in diffs?
   - Trade-offs considered

4. **Demo**:
   - Multi-block example program
   - Show evolution logs
   - Show how blocks evolve independently or together
   - Show final optimized result

### Talking Points

1. **Understanding the problem**:
   - "AlphaEvolve supports multiple blocks, OpenEvolve doesn't"
   - "Infrastructure exists (`parse_evolve_blocks`) but unused"
   - "Current architecture sends full file, needs targeted extraction"

2. **Key challenges**:
   - "Prompt engineering: How much context to show LLM?"
   - "Diff application: Which block does a diff target?"
   - "Reconstruction: Preserving non-evolved code perfectly"

3. **Implementation approach**:
   - "Template-based: Extract blocks, replace with placeholders"
   - "Block-aware diffs: Optional `[BLOCK:name]` markers"
   - "Configurable strategies: All at once vs sequential vs random"

4. **Testing strategy**:
   - "Unit tests for each utility function"
   - "Integration tests for full pipeline"
   - "Real example: Multi-stage data pipeline"

5. **Backward compatibility**:
   - "Opt-in via config flag"
   - "Single-block files work unchanged"
   - "Gradual migration path"

### Questions to Expect

**Q: Why not just use separate files?**
A: "Separate files lose shared context (imports, helpers). Multi-block keeps cohesion while enabling focused evolution. Also mirrors AlphaEvolve's design."

**Q: How do you prevent LLM from modifying non-evolved code?**
A: "Template system preserves non-evolved sections. Diff application validates search text only exists in evolved blocks. Rejects diffs targeting template code."

**Q: What if blocks depend on each other?**
A: "Phase 1: Evolve all blocks together (coordinated). Phase 2: Add explicit dependency declarations. Could also analyze call graphs automatically."

**Q: Performance impact?**
A: "Minimal. Block extraction is O(n) in file length, done once per iteration. Template substitution is string replacement. Main bottleneck is still LLM calls."

**Q: What about other languages (Rust, R)?**
A: "Comment syntax for markers is language-agnostic (`# EVOLVE`, `// EVOLVE`, etc.). Reconstruction is text-based, no parsing needed. Works for any language."

---

## Risks & Mitigation

### Risk 1: LLM doesn't understand block markers
**Mitigation**:
- Clear examples in prompt
- Fallback to implicit detection
- Monitor success rate, iterate on prompt

### Risk 2: Block reconstruction bugs
**Mitigation**:
- Extensive unit tests
- Validation after reconstruction
- Checksum/hash comparison

### Risk 3: Added complexity for limited benefit
**Mitigation**:
- Make feature optional (flag)
- Provide clear use case examples
- Measure: Does multi-block find better solutions?

### Risk 4: Backward compatibility breaks
**Mitigation**:
- Comprehensive testing of single-block mode
- Default to disabled
- Gradual rollout

---

## Success Metrics

### For Implementation
- ✅ All unit tests pass
- ✅ Multi-block example evolves successfully
- ✅ Single-block examples still work
- ✅ Code coverage >80% for new code

### For Implementation
- ✅ Clearly explain the problem and solution
- ✅ Demonstrate working prototype
- ✅ Show understanding of trade-offs
- ✅ Handle questions about edge cases

### For Production
- ✅ Users can create multi-block examples
- ✅ Block-aware diffs work reliably
- ✅ No performance regression
- ✅ Documentation enables self-service adoption

---

## Next Steps

1. **Review this plan** with team
2. **Agree on scope**: Full implementation vs proof-of-concept?
3. **Set up development branch**: `feature/multi-block-evolution`
4. **Start with Phase 1**: Block extraction utilities
5. **Iterate based on feedback**: Adjust approach as needed

---

## Conclusion

This feature extends OpenEvolve to match a key capability of AlphaEvolve: evolving multiple code sections within a single file. The implementation is:

- **Architecturally sound**: Integrates cleanly with existing pipeline
- **Backward compatible**: Opt-in, doesn't break existing examples
- **Well-tested**: Comprehensive unit and integration tests
- **Documented**: Clear examples and user guides
- **Flexible**: Configurable strategies for different use cases

The core insight is that the infrastructure for parsing blocks **already exists** but is unused. This project mainly involves:
1. Integrating the unused `parse_evolve_blocks()` function
2. Building a template-based reconstruction system
3. Extending prompts to communicate block structure to LLM
4. Adding block-aware diff application

**Estimated effort**: 5-7 days for a complete, tested, documented implementation.

This is a substantial, well-scoped project that demonstrates:
- Understanding of complex codebases
- System design skills
- Attention to backward compatibility
- Comprehensive testing practices
- Clear documentation

Good luck 🚀
