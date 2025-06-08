# packages/vlm/gemma_vlm/config.py
from jetson_containers import L4T_VERSION
import os

# Define L4T version-specific dependencies if necessary
# For now, assume common dependencies work across recent L4T versions
# L4T_VERSION_DEPENDENT_REQUIRES = []
# if L4T_VERSION.major >= 36:  # Example for JetPack 6.x
#     L4T_VERSION_DEPENDENT_REQUIRES.append('some_jp6_specific_package')
# else: # Example for JetPack 5.x
#     L4T_VERSION_DEPENDENT_REQUIRES.append('some_jp5_specific_package')

def gemma_vlm_package(model_id='google/gemma-3-4b-it', name_suffix='', default=False, requires_arg=None): # Renamed 'requires' to 'requires_arg' to avoid conflict
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
            'transformers',  # This is a jetson-containers package
            'pytorch',       # This is a jetson-containers package
        ],
        'build_args': {
            'GEMMA_VLM_MODEL_ID': model_id, # Pass model ID to Dockerfile
            'HUGGINGFACE_TOKEN': '${HUGGINGFACE_TOKEN:-None}', # Pass HF token, default to None if not set
        },
        'notes': f"Container for Gemma 3 VLM model: {model_id}. Uses Gemma 3 architecture.",
        'requires': [] 
    }

    if default:
        pkg['alias'] = 'gemma_vlm' # A shorter alias for the default model

    if requires_arg: # Use the renamed argument here
        # Combine base dependencies with any L4T-specific ones
        # Note: 'depends' is for jetson-container package dependencies.
        # 'requires' is typically for system-level things like L4T version.
        # If 'requires_arg' was meant to add to 'depends', that logic would go here.
        # For now, assuming 'requires_arg' might be for the 'requires' key if it were more complex.
        # If it was meant to add to 'depends', it should be:
        # pkg['depends'] = list(set(pkg['depends'] + requires_arg))
        # If it was for the 'requires' key (e.g. L4T version specifier string):
        pkg['requires'] = requires_arg # Or some processing if it's a list for 'requires'
        
    return pkg

# Define one or more specific model packages to be available.
# Default model: google/gemma-3-4b-it
gemma_3_4b_it = gemma_vlm_package(
    model_id='google/gemma-3-4b-it',
    name_suffix='3-4b-it', # Suffix for clarity
    default=True # Make this the default 'gemma_vlm' package
    # requires_arg=L4T_VERSION_DEPENDENT_REQUIRES # Add if specific L4T dependencies are identified
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
