# config for training mymodel (81M) down to loss of 1.90713
# Ran 850,000 iterations on 1 node with 2x A100 40GB
# Ran more 1,550,000 iterations on 1 node with 1x H100 95GB
# For a total of 580B tokens (~ 3x 193B)
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_mymodel.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='mymodel-67M'

init_from = 'resume'

dataset = 'gigaverbo'
# these make the total batch size be ~0.245M
# 120 batch size * 512 block size * 4 gradaccum * 1 GPUs = 245,760
batch_size = 120
block_size = 512
gradient_accumulation_steps = 4 * 1

# this makes total number of tokens be ~193B
max_iters = 2400000
lr_decay_iters = 2400000
learning_rate = 6e-4 # max learning rate
min_lr = 9e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 25000
eval_iters = 500
log_interval = 10

# weight decay
weight_decay = 1e-1
