EXPERIMENTS = {
    # -------------------------
    # CNN 類（顯存安全）
    # -------------------------
    "AICUP_unet3d_base": {
        "enabled": True,
        "vote": True,
        "model_name": "unet3d",
        "params": {
            "patch_size": 2,
            "feature_size": 48,
            "depths": [2, 2, 3, 2],
            "drop_rate": 0.0,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "INSTANCE",
            "roi_z": 112,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },

    "AICUP_attention_unet": {
        "enabled": True,
        "vote": True,
        "model_name": "attention_unet",
        "params": {
            "patch_size": 2,
            "feature_size": 48,
            "depths": [2, 2, 2, 2],
            "drop_rate": 0.0,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "instance",
            "roi_z": 112,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },

    "AICUP_vnet": {
        "enabled": False,
        "vote": False,
        "model_name": "vnet",
        "params": {
            "patch_size": 2,
            "feature_size": 32,
            "depths": [2, 2, 2, 2],
            "drop_rate": 0.0,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "instance",
            "roi_z": 112,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },

    "AICUP_dynunet": {
        "enabled": True,
        "vote": True,
        "model_name": "DynUNet",
        "params": {
            "patch_size": 2,
            "feature_size": 32,
            "depths": [2, 2, 2, 2],
            "drop_rate": 0.0,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "instance",
            "roi_z": 112,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },

    # -------------------------
    # Transformer 類（小心顯存）
    # -------------------------
    "AICUP_unetr_light": {
        "enabled": False,
        "vote": True,
        "model_name": "unetr",
        "params": {
            "patch_size": 16,
            "feature_size": 32,
            "depths": [2, 2, 2, 2],
            "drop_rate": 0.1,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "layer",
            "roi_z": 112,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "max_epoch": 300,
        },
    },

    "AICUP_swinunetr": {
        "enabled": True,
        "vote": True,
        "model_name": "swinunetr",
        "params": {
            "patch_size": 4,        # ← 修正！
            "feature_size": 32,
            "depths": [2, 2, 2, 2],
            "drop_rate": 0.1,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "instance",
            "roi_z": 96,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },

    "AICUP_unetr_pp": {
        "enabled": False,
        "vote": False,
        "model_name": "unetr_pp",
        "params": {
            "patch_size": 16,
            "feature_size": 32,
            "depths": [2, 3, 3, 2],
            "drop_rate": 0.1,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "instance",
            "roi_z": 112,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },

    "AICUP_uxnet_small": {
        "enabled": True,
        "vote": True,
        "model_name": "uxnet",
        "params": {
            "patch_size": 2,
            "feature_size": 24,
            "depths": [2, 2, 2, 2],
            "drop_rate": 0.1,
            "kernel_size": 3,
            "exp_rate": 4,
            "norm_name": "instance",
            "roi_z": 96,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "max_epoch": 300,
        },
    },
}
