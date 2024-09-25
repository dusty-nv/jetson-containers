SAPIENS_LITE_MODELS_URL = {
    "depth": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-depth-0.6b-torchscript",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-depth-1b-torchscript",
        "sapiens_2b": "https://huggingface.co/facebook/sapiens-depth-2b-torchscript"
    },
    "detector": {},
    "normal": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-normal-0.6b-torchscript",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-normal-1b-torchscript",
        "sapiens_2b": "https://huggingface.co/facebook/sapiens-normal-2b-torchscript"
    },
    "pose": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-pose-1b-torchscript",
    },
    "seg": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-seg-0.6b-torchscript",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-seg-1b-torchscript",
    }
}

SAPIENS_LITE_MODELS_PATH = {
    "depth": {
        "sapiens_0.3b": "checkpoints/depth/sapiens-depth-0.3b-torchscript",
        "sapiens_0.6b": "checkpoints/depth/sapiens-depth-0.6b-torchscript",
        "sapiens_1b": "checkpoints/depth/sapiens-depth-1b-torchscript",
        "sapiens_2b": "checkpoints/depth/sapiens-depth-2b-torchscript"
    },
    "detector": {},
    "normal": {
        "sapiens_0.3b": "checkpoints/normal/sapiens-normal-0.3b-torchscript",
        "sapiens_0.6b": "checkpoints/normal/sapiens-normal-0.6b-torchscript",
        "sapiens_1b": "checkpoints/normal/sapiens-normal-1b-torchscript",
        "sapiens_2b": "checkpoints/normal/sapiens-normal-2b-torchscript"
    },
    "pose": {
        "sapiens_0.3b": "checkpoints/pose/sapiens-pose-0.3b-torchscript.pt2",
        "sapiens_0.6b": "checkpoints/pose/sapiens-pose-0.6b-torchscript.pt2",
        "sapiens_1b": "checkpoints/pose/sapiens-pose-1b-torchscript",
    },
    "seg": {
        "sapiens_0.3b": "checkpoints/seg/sapiens-seg-0.3b-torchscript",
        "sapiens_0.6b": "checkpoints/seg/sapiens-seg-0.6b-torchscript",
        "sapiens_1b": "checkpoints/seg/sapiens-seg-1b-torchscript",
    }
}

LABELS_TO_IDS = {
    "Background": 0,
    "Apparel": 1,
    "Face Neck": 2,
    "Hair": 3,
    "Left Foot": 4,
    "Left Hand": 5,
    "Left Lower Arm": 6,
    "Left Lower Leg": 7,
    "Left Shoe": 8,
    "Left Sock": 9,
    "Left Upper Arm": 10,
    "Left Upper Leg": 11,
    "Lower Clothing": 12,
    "Right Foot": 13,
    "Right Hand": 14,
    "Right Lower Arm": 15,
    "Right Lower Leg": 16,
    "Right Shoe": 17,
    "Right Sock": 18,
    "Right Upper Arm": 19,
    "Right Upper Leg": 20,
    "Torso": 21,
    "Upper Clothing": 22,
    "Lower Lip": 23,
    "Upper Lip": 24,
    "Lower Teeth": 25,
    "Upper Teeth": 26,
    "Tongue": 27,
}