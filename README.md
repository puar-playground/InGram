# InGram: Inductive Knowledge Graph Embedding via Relation Graphs
This code is the official implementation of the following [paper](https://proceedings.mlr.press/v202/lee23c.html):


## Requirements
We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.
You can install all requirements with:
```shell
pip install -r requirements.txt
```

## Reproducing the Reported Results
Place the unzipped ckpt folder in the same directory with the codes. You can download the checkpoints from https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing.
The command to reproduce the results in our paper:
```python
python3 test.py --best --data_name [dataset_name]
```

## Training from Scratch

To train InGram from scratch, run `train.py` with arguments. Please refer to `my_parser.py` for the examples of the arguments. Please tune the hyperparameters of our model using the range provided in Appendix C of the paper because the best hyperparameters may be different due to randomness.

The list of arguments of `train.py`:
- `--data_name`: name of the dataset
- `--exp`: experiment name
- `-m, --margin`: $\gamma$
- `-lr, --learning_rate`: learning rate
- `-nle, --num_layer_ent`: $\widehat{L}$
- `-nlr, --num_layer_rel`: $L$
- `-d_e, --dimension_entity`: $\widehat{d}$
- `-d_r, --dimension_relation`: $d$
- `-hdr_e, --hidden_dimension_ratio_entity`: $\widehat{d'}/\widehat{d}$
- `-hdr_r, --hidden_dimension_ratio_relation`: $d'/d$
- `-b, --num_bin`: $B$
- `-e, --num_epoch`: number of epochs to run
- `--target_epoch`: the epoch to run test (only used for test.py)
- `-v, --validation_epoch`: duration for the validation
- `--num_head`: $\widehat{K}=K$
- `--num_neg`: number of negative triplets per triplet
- `--best`: use the provided checkpoints (only used for test.py)
- `--no_write`: don't save the checkpoints (only used for train.py)

## Task
The model file is in `model.py`. For a KG with $x$ entities and $y$ relations, if the embedding dimension for entity and relation were set to be $d_e$ and $d_r$. (They use 32 for these dimensions) The input is random initialized embeddings of dimension [$x$, $d_e$] and [$y$, $d_r$]. The outputs are learned embeddings with the same dimension. I need 2 modifications on the code.

### 1. Model for batched input
A new `model.py` file. The model should take batched input. For example, a batch size of $3$. The initialized inputs should have a shape: [3, $x$, $d_e$] and [3, $y$, $d_r$]. Please make sure each batch is initialized by the method in `initialize.py`. And make sure every step in the modified forward function is differentiable, (allow torch to do auto grad). 

1. move the `self.rel_proj` layer in the `score` function in `model.py` line $1$


### 2. Embedding refinement function
A function that take a batch of KG embeddings (batch outputs of a trained InGram model) as inputs. And outputs refined embeddings in the same shape. 
1. You can run inference $N$ times and save the embeddings as [$N$, $x$, $d_e$] and [$N$, $y$, $d_r$] to be the input of this function.
2. Inside the function, you first define all input as `torch.nn.Parameter`
3. Then you need to define an optimizer to update all embeddings to maximize the loss function.
4. The loss function is the `score` defined in `model.py`. And for batch size $N$, you should have $N$ losses. And the total loss is the mean of these $N$ losses. 
5. Repeat the optimization loop for $t$ (a hyper-parameter of the function) times.
6. Use `detach()` to delete the gradiant of all embeddings. The output should be torch tensors without gradient and with shapes of [$N$, $x$, $d_e$] and [$N$, $y$, $d_r$].





