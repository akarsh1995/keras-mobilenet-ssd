config = {
    "model": {
        "name": "ssd_mobilenetv2",
        "input_size": 300,
        "extra_box_for_ar_1": True,
        "l2_regularization": 0.0005,
        "kernel_initializer": "he_normal",
        "width_multiplier": 0.5,
        "default_boxes": {
            "variances": [
                0.1,
                0.1,
                0.2,
                0.2
            ],
            "min_scale": 0.2,
            "max_scale": 0.9,
            "layers": [
                {
                    "name": "block_13_expand_relu",
                    "size": 19,
                    "offset": [
                        0.5,
                        0.5
                    ],
                    "aspect_ratios": [
                        2.0,
                        3.0
                    ]
                },
                {
                    "name": "block_16_project_BN",
                    "size": 10,
                    "offset": [
                        0.5,
                        0.5
                    ],
                    "aspect_ratios": [
                        2.0,
                        3.0
                    ]
                },
                {
                    "name": "conv17_2/relu",
                    "size": 5,
                    "offset": [
                        0.5,
                        0.5
                    ],
                    "aspect_ratios": [
                        2.0,
                        3.0
                    ]
                },
                {
                    "name": "conv18_2/relu",
                    "size": 3,
                    "offset": [
                        0.5,
                        0.5
                    ],
                    "aspect_ratios": [
                        2.0,
                        3.0
                    ]
                },
                {
                    "name": "conv19_2/relu",
                    "size": 2,
                    "offset": [
                        0.5,
                        0.5
                    ],
                    "aspect_ratios": [
                        2.0,
                        3.0
                    ]
                },
                {
                    "name": "conv20_2/relu",
                    "size": 1,
                    "offset": [
                        0.5,
                        0.5
                    ],
                    "aspect_ratios": [
                        2.0,
                        3.0
                    ]
                }
            ]
        }
    },
    "training": {
        "match_threshold": 0.5,
        "neutral_threshold": 0.3,
        "min_negative_boxes": 0,
        "negative_boxes_ratio": 3,
        "alpha": 1
    }
}
