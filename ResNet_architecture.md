```mermaid
graph TD
    subgraph ResNetUFC General Architecture
        %% Inputs
        InputCat[Categorical Inputs<br>Indices] --> Embs
        InputNum[Numeric Inputs<br>Features] --> NumBN[Batch Normalization]
        
        %% Preprocessing Stage
        subgraph Embedding Layer Parallel
            Embs[Embedding Lookups<br>Multiple layers concatenated]
        end
        
        Embs --> Concat[Concatenate]
        NumBN --> Concat
        
        %% Initial Projection
        Concat --> InitFC[Linear Projection<br>To: H_0]
        InitFC --> InitBN[Batch Norm]
        InitBN --> InitReLU[ReLU]
        
        %% Residual Stage
        subgraph Residual Stage 1
            InitReLU --> ResBlockStart
            
            subgraph Residual Block
                ResBlockStart[Input] --> RB_FC1[Linear<br>H_0 -> H_0]
                RB_FC1 --> RB_BN1[Batch Norm]
                RB_BN1 --> RB_ReLU1[ReLU]
                RB_ReLU1 --> RB_Drop[Dropout p=0.3]
                RB_Drop --> RB_FC2[Linear<br>H_0 -> H_0]
                RB_FC2 --> RB_BN2[Batch Norm]
                
                %% Skip Connection
                ResBlockStart -- Identity Skip Connection --> RB_Add((+))
                RB_BN2 --> RB_Add
                RB_Add --> RB_FinalReLU[ReLU]
            end
            
            %% Transition Block
            subgraph Transition Block
                RB_FinalReLU --> TransFC[Linear Projection<br>H_0 -> H_1]
                TransFC --> TransBN[Batch Norm]
                TransBN --> TransReLU[ReLU]
            end
        end
        
        %% Output Head
        subgraph Output Head
            TransReLU --> HeadDrop[Dropout p=0.2]
            HeadDrop --> HeadFC[Linear<br>H_1 -> 1]
            HeadFC --> HeadSig[Sigmoid]
        end
        
        HeadSig --> FinalOutput[Output Probability]
    end

    style InputCat fill:#f9f,stroke:#333,stroke-width:2px
    style InputNum fill:#f9f,stroke:#333,stroke-width:2px
    style FinalOutput fill:#f9f,stroke:#333,stroke-width:2px
    style Concat fill:#ffd,stroke:#333
    style Embs fill:#ded,stroke:#333
    style ResBlockStart fill:#fff,stroke:none
    style RB_Add fill:#fff,stroke:#333
    style InitFC fill:#def,stroke:#333
    style RB_FC1 fill:#def,stroke:#333
    style RB_FC2 fill:#def,stroke:#333
    style TransFC fill:#def,stroke:#333
    style HeadFC fill:#def,stroke:#333