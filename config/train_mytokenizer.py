# config for training mymodel (67M) down to very nice loss of ???
# Ran more 1,725,000 iterations on 1 node with 1x H100 95GB
# For a total of 420B tokens (~ 3x 140B)
# launch as the following (e.g. in a screen session) and wait ~5 days:
# python train.py config/train_mytokenizer.py

wandb_log = True
wandb_project = 'mytokenizer'
wandb_run_name='mymodel-67M'

init_from = 'scratch'

dataset = 'mytokenizer'
# these make the total batch size be ~0.245M
# 120 batch size * 512 block size * 4 gradaccum * 1 GPUs = 245,760
batch_size = 120
block_size = 512
gradient_accumulation_steps = 4 * 1

# this makes total number of tokens be ~193B
max_iters = 1725000
lr_decay_iters = 1725000
learning_rate = 6e-4 # max learning rate
min_lr = 9e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 25000
eval_iters = 500
log_interval = 10

# weight decay
weight_decay = 1e-1
