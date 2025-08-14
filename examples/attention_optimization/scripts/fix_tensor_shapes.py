import re

# Read the file
with open('./mlir/self_attn_with_consts_linalg_dialect.mlir', 'r') as f:
    content = f.read()

# Pattern to match tensor.expand_shape without output_shape
pattern = r'tensor\.expand_shape\s+([^[]+)\s+(\[\[.*?\]\])\s*:\s*([^)]+)\s+into\s+(tensor<[^>]+>)'

def add_output_shape(match):
    var, indices, input_type, output_type = match.groups()
    
    # Extract dimensions from output tensor type
    dims_match = re.search(r'tensor<([^>]+)>', output_type)
    if dims_match:
        dims_str = dims_match.group(1)
        # Extract just the dimension numbers (ignore 'xf32' etc.)
        dims = re.findall(r'\d+', dims_str.split('x')[:-1])  # Exclude the type part
        if dims:
            output_shape = '[' + ', '.join(dims) + ']'
            return f'tensor.expand_shape {var} {indices} output_shape {output_shape} : {input_type} into {output_type}'
    
    return match.group(0)  # Return original if we can't parse

# Apply the fix
fixed_content = re.sub(pattern, add_output_shape, content, flags=re.MULTILINE)

# Write back
with open('./mlir/self_attn_with_consts_linalg_dialect.mlir', 'w') as f:
    f.write(fixed_content)

print("Fixed tensor.expand_shape syntax")
