# packages/vlm/gemma_vlm/config.py
from jetson_containers import L4T_VERSION

# Define L4T version-specific dependencies if necessary
# For now, assume common dependencies work across recent L4T versions
# L4T_VERSION_DEPENDENT_REQUIRES = []
# if L4T_VERSION.major >= 36:  # Example for JetPack 6.x
#     L4T_VERSION_DEPENDENT_REQUIRES.append('some_jp6_specific_package')
# else: # Example for JetPack 5.x
#     L4T_VERSION_DEPENDENT_REQUIRES.append('some_jp5_specific_package')

def gemma_vlm_package(model_id='google/gemma-3-4b-it', name_suffix='', default=False, requires=None):
    """
    Generates a package configuration for a Gemma 3 VLM model.
    """
    # Generate a package name, using the model_id's last part if no suffix is provided
    if not name_suffix:
        name_suffix = model_id.split("/")[-1] # e.g. "gemma-3-4b-it"
        
    pkg_name = f'gemma_vlm:{name_suffix}'
    
    pkg = {
        'name': pkg_name,
        'dockerfile': 'Dockerfile',
        'test': 'test_gemma_vlm.py', # Script to run for testing this package
        'depends': [
            'transformers',  # Ensure this pulls a recent enough version for Gemma3
            'pytorch',       # PyTorch is a core dependency
            'pillow',        # For image processing
            'accelerate',    # Hugging Face Accelerate for efficient loading
            'requests',      # For the test script to download a sample image
            'einops',        # Often a dependency for vision-language models
            'bitsandbytes',  # For potential future 8-bit/4-bit quantization
        ],
        'build_args': {
            'GEMMA_VLM_MODEL_ID': model_id, # Pass model ID to Dockerfile
            'HUGGINGFACE_TOKEN': '${HUGGINGFACE_TOKEN:-None}', # Pass HF token, default to None if not set
        },
        'notes': f"Container for Gemma 3 VLM model: {model_id}. Uses Gemma 3 architecture."
    }

    if default:
        pkg['alias'] = 'gemma_vlm' # A shorter alias for the default model

    if requires:
        # Combine base dependencies with any L4T-specific ones
        pkg['depends'] = list(set(pkg['depends'] + requires))
        
    return pkg

# Define one or more specific model packages to be available.
# Default model: google/gemma-3-4b-it
gemma_3_4b_it = gemma_vlm_package(
    model_id='google/gemma-3-4b-it',
    name_suffix='3-4b-it', # Suffix for clarity
    default=True # Make this the default 'gemma_vlm' package
    # requires=L4T_VERSION_DEPENDENT_REQUIRES # Add if specific L4T dependencies are identified
)

# Example for another Gemma 3 model variant (e.g., a different size or fine-tune)
# gemma_3_xb_it = gemma_vlm_package(
#     model_id='google/gemma-3-xb-it', # Replace xb with actual size like 2b or 9b
#     name_suffix='3-xb-it'
# )

# The 'package' variable is what jetson-containers build system will discover.
package = [
    gemma_3_4b_it,
    # gemma_3_xb_it, # Uncomment to add more variants to the build system
]
