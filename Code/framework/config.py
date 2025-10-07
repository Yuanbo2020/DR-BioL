import os

######
win_size = 200
step_size = int(win_size/2)
mel_bins = 64  # VAE: 2 s: (8, 50, 16)
min_duration = step_size  # 100  # 100 frames are 1s
MEL_feature_channel = 1

all_species_labels = ['an arabiensis', 'culex pipiens complex', 'ae aegypti', 'an funestus ss',
                      'an squamosus', 'an coustani', 'ma uniformis', 'aedes albopictus']
all_domain_labels = ['D1', 'D2', 'D3', 'D4']


endswith = '.pth'
early_stopping_max_overrun = 10
warmup_epoch = 50





