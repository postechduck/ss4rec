# SS4Rec
## Continuous-Time Sequential Recommendation with State Space Models (Under review)
![image](https://github.com/Blank141/SS4Rec/blob/main/ss4rec.png)
## Requirements
```
s5-pytorch
mamba-ssm
causal-conv1d
recbole 1.0
torch
```

## Usage
Firstly, replacing recbole.data.sequential_dataset.py with SS4Rec/sequential_dataset.py
```
python run.py
```

### Other Related Projects
The code repository references [RecBole]https://github.com/RUCAIBox/RecBole and [Mamba4Rec]https://github.com/chengkai-liu/Mamba4Rec.
Thanks a lot for their work!
