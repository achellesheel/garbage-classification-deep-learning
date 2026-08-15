"""Maps each source dataset's folder names to the unified 7-class taxonomy.

Unified classes: cardboard, glass, metal, paper, plastic, trash, biological
battery/clothes/shoes are intentionally dropped (only in 2 of 3 datasets).
"""

UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "biological"]

# Each entry: source folder root, and its class-name -> unified-class map.
# A unified value of None means "drop this source class".
SOURCES = [
    {
        "name": "trashnet",
        "root": "trashnet/dataset-resized",
        "class_map": {
            "cardboard": "cardboard",
            "glass": "glass",
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "trash": "trash",
        },
    },
    {
        "name": "garbage12",
        "root": "garbage-classification/garbage_classification",
        "class_map": {
            "cardboard": "cardboard",
            "green-glass": "glass",
            "brown-glass": "glass",
            "white-glass": "glass",
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "trash": "trash",
            "biological": "biological",
            "battery": None,
            "clothes": None,
            "shoes": None,
        },
    },
    {
        "name": "garbagev2",
        "root": "garbage-classification-v2/original",
        "class_map": {
            "cardboard": "cardboard",
            "glass": "glass",
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "trash": "trash",
            "biological": "biological",
            "battery": None,
            "clothes": None,
            "shoes": None,
        },
    },
]
