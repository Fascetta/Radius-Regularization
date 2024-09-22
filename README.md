# CPHNN: Calibration Properties of Hyperbolic Neural Networks
## Introduction
This repository serves as the investigation for the Bachelor's final thesis in Applied Computer Science and Artificial Intelligence. The focus of this thesis is on studying the calibration properties of hyperbolic neural networks.


## Objective
The aim of this thesis study is to investigate the calibration properties of hyperbolic neural networks.

## Investigation
1. Replication of Results
We will replicate the results from the experiment described in the provided repository, specifically focusing on the **Fully Hyperbolic Convolutional Neural Network** based on the **Lorentz model**.
https://github.com/kschwethelm/HyperbolicCV

2. Calibration Property and Correlation Analysis
Once the replication is completed, we will proceed to investigate the calibration properties of the network. This will involve:

* Computing the Expected Calibration Error (ECE) scores.
https://openreview.net/pdf?id=QRBvLayFXI
* Correlating the ECE scores with the hyperbolic radius.

## Additional Resources
For further understanding of the hyperbolic radius and its implications, please refer to the provided paper.
https://arxiv.org/pdf/2310.08390

## Conclusion
By conducting this investigation, we aim to contribute to the understanding of calibration properties in hyperbolic neural networks, shedding light on their effectiveness and potential applications.

``` mermaid
flowchart LR
    %% Model Architecture Subgraph %%
    subgraph ModelArchitecture [Model Architecture]
        style ModelArchitecture fill:#E7F5FF,stroke:#007ACC,stroke-width:2px
        subgraph Architecture
            style Architecture fill:#FFD9E7,stroke:#C71585,stroke-width:2px
            A1[ResNet18]
            A2[ResNet50]  
        end
        subgraph Optimizer
            style Optimizer fill:#D8F8B7,stroke:#2D7F28,stroke-width:2px
            O1[Adam]
            O2[SGD]
        end
        subgraph LR_Scheduler
            style LR_Scheduler fill:#FFF3BA,stroke:#F4D03F,stroke-width:2px
            L1[Learning Rate]
            L2[LR Scheduler Milestones]
            L3[LR Scheduler Gamma]
        end
        subgraph Criterion
            style Criterion fill:#FFF0C7,stroke:#FFA500,stroke-width:2px
            C1[CrossEntropyLoss]
            C2[RadiusLoss]
        end
    end

    %% Dataset Subgraph %%
    subgraph Dataset
        style Dataset fill:#F7F3E3,stroke:#9C640C,stroke-width:2px
        D1[(MNIST)]
        D2[(CIFAR10)]   
        D3[(CIFAR100)]
        D4[(TinyImageNet)]
    end

    %% Radius Loss %%
    subgraph RadiusLoss [Radius Loss]
        style RadiusLoss fill:#D6EAF8,stroke:#3498DB,stroke-width:2px
        direction LR
        CoC[[Correct Counts]] & TC[[Total Counts]] --> DIV
        style DIV fill:#AED6F1,stroke:#21618C
        DIV((Division)) --> CC((Class Confidence))
        R[[Radii]] & CC --> C3((MSE))
        C3 --> RL0[Radius Loss]
    end

    %% Training Loop Subgraph %%
    subgraph TL [Training Loop]
        style TL fill:#EBF5FB,stroke:#1F618D,stroke-width:2px
        direction LR
        B((Batch)) --> I2
        B --|Input|--> M[/Model/]
        M --> P1[Prediction]
        I2((Label)) & P1 --> RL1(Radius Loss) & CEL(Cross Entropy Loss)

        CEL & RL1 --> P[Sum Loss]
        style P fill:#F9E79F,stroke:#D4AC0D,stroke-width:2px
        P --> G4["Backpropagation"]
        G4 --> G5["Weight Update"]
        G5 --> G6["Update Metrics"]
        G6 -->|Next Batch| B
    end

    %% Output Subgraph %%
    subgraph Evaluation
        style Evaluation fill:#FDEBD0,stroke:#CA6F1E,stroke-width:2px
        subgraph Accuracy
            style Accuracy fill:#E8DAEF,stroke:#8E44AD,stroke-width:2px
            T1[Acc1]
            T5[Acc5]
        end
        subgraph Calibration
            style Calibration fill:#F5EEF8,stroke:#9B59B6,stroke-width:2px
            E1[[ECE]]
            E2[[MCE]]
            E3[[RMSCE]]
        end
    end

    %% Connections %%
    Dataset --> B
    P1 & I2 ---> Evaluation
```
