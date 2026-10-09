from enum import Enum
from .defaults import GPU_SPECS, MODEL_SPECS, DATASET_SPECS


class Models(Enum):
    Mixer = "mixer"
    Resmlp = "resmlp"
    Asmlp = "asmlp"
    DistilBERT = "DistilBERT"
    Transformer = "Transformer"
    ViT = "ViT"
    DeiTTiny = "DeiT-Tiny"
    ResNet50 = "ResNet50"
    VGG16 = "VGG16"
    MobileNetV2 = "MobileNetV2"
    DenseNet121 = "DenseNet121"
    
    def __str__(self):
        return self.value
    
    @property
    def specs(self):
        return MODEL_SPECS.get(self.value, {})


class Datasets(Enum):
    Cifar100 = "CIFAR100"
    Cifar10 = "CIFAR10"
    TinyImageNet = "TinyImageNet"
    STL10 = "STL10"
    SST2 = "SST2"
    IMDB = "IMDB"
    CNNDM_SUM = "CNNDM_SUM"
    WMT14_EN_DE = "WMT14_EN_DE"

    def __str__(self):
        return self.value
    
    @property
    def specs(self):
        return DATASET_SPECS.get(self.value, {})


class Optimizers(Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    SGD = "SGD"
    RMSProp = "rmsprop"
    Adafactor = "Adafactor"

    def __str__(self):
        return self.value


class Devices(Enum):
    A100 = "A100"
    V100 = "V100"
    T4 = "T4"
    H100 = "H100"
    A40 = "A40"
    RTX5090 = "RTX5090"
    L4 = "L4"
    L40S = "L40S"
    P4 = "P4"
    P100 = "P100"

    def __str__(self):
        return self.value
    
    @property
    def specs(self):
        return GPU_SPECS.get(self.value, {})