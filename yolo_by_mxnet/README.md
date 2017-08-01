DLMX: DLVC Detection Library with MXNet Backend
----------------------------------------

## File System Level Introduction

The tree structure of **DLMX**:

```
.
├── dataset
│   └── voc_label.py
│   └── make_list.py
└── utli
│   └── ipp.py
└── model
│   └── inception-bn
└── examples
│   └── yolo
│       └── data.py
│       └── metric.py
│       └── symbol.py
├── README.md
└── requirements.txt

```

+ DLVC_root
    - dataset: 
    Contains helper functions to load common databases
        - voc_label.py
        - make_list.py
    - utlis: 
    Contains most common used detection related helper functions
        - ipp.py
    - model:
    Contains some pre-trained models
        - inception-bn
    - exampes:
        - yolo
    - tools: 
    Coming soon...
