# mechanistic-interpretability

`mechanistic-interpretability` is a toolbox for faciliting mechanistic interpretability (MI) research on small models and toy tasks.


The goal intially of this project is to provide an easy-to-use framework of models, toy tasks, and tools for training, evaluating, and analyzing models through the lens of mechanistic interpretability.


The core of the framework is built around PyTorch [2] and PyTorch Lightning [3] to reduce boilerplate code, provide modularity and flexibility, and to design around the APIs of popular well-documented libraries.


The repo also utilizes Hydra [4] for configuration of experiments, but it can be used without it. Tensorboard is also optionally used, but can be replaced (or removed) with your own logging/experiment tracking tools by using different PyTorch Lightning loggers.

## Get started
### Modular Addition
Modular addition is the main task studied in "A Mechanistic Interpretability Analysis of Grokking" by Neel Nanda [1]. It's a toy task where we train an autoregressive decoder-only transformer to learn:

`x + y = (x + y) % p`
 
where the input to the model is:
 
`x + y =`

and the target is defined by

`(x + y) % p`
 
`p` is whatever constant we choose (i.e. in [1] they use the p=113). We implement this by forming an input sequence to be three tokens:
 
`x | y | = `
  
where `|` is used only here to illustrate the tokenization and a target sequence of:
  
`-1 | -1 | (x + y) % p`
   
where `-1` is a reserved value that the loss ignores since we only care about learning to predict the last token.

We can reproduce the training run in [1] by running the module addition task from the CLI
```
$ python main.py task=modular_addition
```

The loss curves of the original are pretty accurately reproduced.

Original [1]          |  This repo
:-------------------------:|:-------------------------:
<img src=assets/original-modadd.JPG width="1000"> |  <img src=assets/reproduce-modadd.JPG width="1000">

The task parameters (data, model, and training hyperparams) can be modified directly in `configs/task/modular_addition.yaml` or through Hydra's CLI override grammar. For example if we wanted to specify a different value for `p` or use another activation function in our model we can do that with the following:

```
$ python main.py task=modular_addition datamodule.p=42 model.model.act_type=gelu
```

### N-digit addition
N-digit addition is a toy task where we train an autoregressive decoder-only transformer to learn to add two numbers together digit by digit. We implement this by forming a sequence where each digit of the two numbers and the sum are a token. i.e. for the problem of 42 + 73 = 115 the input at training time would be encoded as ` 4 | 2 | 7 | 3 | 1 | 1 ` while the targets are ` -1 | -1 | -1 | 1 | 1 | 5 ` (where `-1` is a reserved value that the loss ignores).

This repo only implements 2 and 3 digit addition (code taken from Andrej Karpathy's minGPT [5]), while [1] showed results on 5 digit addition (on the todo list for this repo).

The training for this task can be invoked from the command line with:
```
$ python main.py task=3_digit_addition
```

The n-digit addition module logs the per-token loss (in addition to the overall loss) to allow for better understanding of the training dynamics as the model learns to do addition.

<img src=assets/per-token-loss-digits.png width="500">

There is also the option to encode the sum in reverse order (the default behavior in minGPT) by setting `reverse_sum = True` in `DigitAdditionDataModule` or using Hydra CLI overrides like such:

```
$ python main.py task=3_digit_addition datamodule.reverse_sum=true
```

### Superposition
There is also support for "Toy Models of Superposition" by Elhage et al. [6] where a small ReLU network is trained to compress and then reconstruct synthetic data in order to investigate how features are represented when there are less dimensions than features.

The training to create the intro figure of the paper can be invoked from the command line with:
```
$ python main.py task=superposition
```
<img src=assets/superposition-intro.png width="1500">

(NOTE: this command does not automatically generate the plot. I used a checkpoint to generate it along with the plotting source code [7] in a notebook.)

## References

[1] https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20#scrollTo=BhhJmRH8IIvy

[2] https://pytorch.org/

[3] https://www.pytorchlightning.ai/

[4] https://hydra.cc/

[5] https://github.com/karpathy/minGPT

[6] https://transformer-circuits.pub/2022/toy_model/index.html

[7] https://github.com/anthropics/toy-models-of-superposition