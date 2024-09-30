SAPIENS_LITE_MODELS_URL = {
    "depth": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript/resolve/main/sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-depth-0.6b-torchscript/resolve/main/sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-depth-1b-torchscript/resolve/main/sapiens_1b_render_people_epoch_88_torchscript.pt2",
        "sapiens_2b": "https://huggingface.co/facebook/sapiens-depth-2b-torchscript/resolve/main/sapiens_2b_render_people_epoch_25_torchscript.pt2"
    },
    "detector": {},
    "normal": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript/resolve/main/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-normal-0.6b-torchscript/resolve/main/sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        "sapiens_2b": "https://huggingface.co/facebook/sapiens-normal-2b-torchscript/resolve/main/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2"
    },
    "pose": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_575_torchscript.pt2",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_600_torchscript.pt2",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2",
    },
    "seg": {
        "sapiens_0.3b": "https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
        "sapiens_0.6b": "https://huggingface.co/facebook/sapiens-seg-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
        "sapiens_1b": "https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }
}

SAPIENS_LITE_MODELS_PATH = {
    "depth": {
        "sapiens_0.3b": "checkpoints/depth/sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
        "sapiens_0.6b": "checkpoints/sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
        "sapiens_1b": "checkpoints/sapiens_1b_render_people_epoch_88_torchscript.pt2",
        "sapiens_2b": "checkpoints/sapiens_2b_render_people_epoch_25_torchscript.pt2"
    },
    "detector": {},
    "normal": {
        "sapiens_0.3b": "checkpoints/normal/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        "sapiens_0.6b": "checkpoints/normal/sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        "sapiens_1b": "checkpoints/normal/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        "sapiens_2b": "checkpoints/normal/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2"
    },
    "pose": {
        "sapiens_0.3b": "checkpoints/pose/sapiens_0.3b_goliath_best_goliath_AP_575_torchscript.pt2",
        "sapiens_0.6b": "checkpoints/pose/sapiens_0.6b_goliath_best_goliath_AP_600_torchscript.pt2",
        "sapiens_1b": "checkpoints/pose/sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2",
    },
    "seg": {
        "sapiens_0.3b": "checkpoints/seg/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
        "sapiens_0.6b": "checkpoints/seg/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
        "sapiens_1b": "checkpoints/seg/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
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