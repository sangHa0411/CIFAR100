# CIFAR100

## Modules
```
|-- Data
|   |-- cifar-100-python
|   |   |-- file.txt~
|   |   |-- meta
|   |   |-- test
|   |   `-- train
|   `-- cifar-100-python.tar.gz
|-- Log
|-- Model
|   `-- resnet_cifar100.pt
|-- model.py
`-- train.py
```

## Model 
  1. ResNet 34-layers
  
## Model Specification
  1. Total layer size : 34
  2. Input image size : 224
  3. Layer size list : [3, 4, 6, 3]
  4. Channel size list : [64, 128, 256, 512]
  5. Kernal size : 3
  6. Class size : 100

## Training 
  1. Optimizer : SGD (momentum = 0.9)
  2. Scheudler : CosineAnnealingLR
      * Init learning rate : 2.5e-4
      * Mininum learning rate : 1e-7
      * Max iteration : 5
  3. Epochs : 50
  4. Batch size : 128

## Data 
  1. CIFAR100

## Reference
  1. Resnet : https://arxiv.org/pdf/1512.03385.pdf

