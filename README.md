# CS-6220-Project

Holistic DL optimization framework

## Dependencies

- Ray
- LRBench

## How to use

### 1. Create custom neural network

For example, Convolutional NN

### 2. Create ConfigurableNet object

### 3. Create config(Python dictionary)

You can use functions inside ray.tune module by calling ConfigurableNet object's tune variable

- lr: initial learning rate
- lrBench: lrbench parameters
- stop_iteration: max iteration
- user_option: user custom option
  - accuracy_threshold: minimum accuracy to pass over. with this option on, shortest runtime config will be the best config

For lrPolicy FIX, do not pass in k0 value as it will be covered by initial learning rate, lr.

### 4. Set search space `set_space()`

You need to pass in:

- dl: DL Framework to use
- net: Custom net created in step 1
- config: Config created in step 3
- optim: DL Optimizer to use

### 5. Load data `data_loader()`

You need to pass in:

- dataloader: dataloader for specific DL framework
- dataset: dataset for specific DL framework
- transforms: transforms for specific DL framework
- data_path: path to data

### 6. Run `run()`

Enjoy exploring!
