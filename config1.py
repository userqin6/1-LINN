# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.8
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 4
lamda_guide = 2
lamda_low_frequency = 1
lamda_like_gauss = 1e-9
device_ids = [0]

#glow
n_flow=1
n_block=1


# Train:
""" 
10，896
""" 
batch_size = 8
cropsize = 256
betas = (0.5, 0.999)
weight_step = 60
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 50
data_name_test = 'DIV2K_valid_HR'
IMAGE_PATH = '/mnt/vdb/Hiding-images-within-images/results/images'
save_processed_img=True
# Dataset
TRAIN_PATH = '/opt/mydownload/train_image/DIV2K_train_HR'
VAL_PATH = '/opt/mydownload/train_image/DIV2K_valid_HR'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/opt/mydownload/model_2/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = '/opt/mydownload/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suf = 'png'
suffix = '/opt/mydownload/model_2/model_best.pt'
suffix1 = '/opt/mydownload/model_2/model.pt'
tain_next = False
trained_epoch = 0
