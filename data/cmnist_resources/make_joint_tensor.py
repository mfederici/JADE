import torch
import os

# Conditional probability distributions are represented as tensors
p_e = {}
p_y_d =torch.zeros([2, 2, 3, 2])
p_e['train'] = torch.zeros([2, 2, 3, 2])
p_e['env_1'] = torch.zeros([2, 2, 3, 2])
p_e['env_2'] = torch.zeros([2, 2, 3, 2])
p_e['env_3'] = torch.zeros([2, 2, 3, 2])
p_c_ye = torch.zeros([2, 2, 3, 2])

# Label
# p(y=0|d=0) = p(y=1|d=1) = 0.75
p_y_d[0, 0] = p_y_d[1, 1] = 0.75
# p(y=1|d=0) = p(y=1|d=0) = 0.25
p_y_d[0, 1] = p_y_d[1, 0] = 0.25

# Environment

# Train
# p(e=0)=p(e=1) = 0.5
p_e['train'][:, :, 0] = p_e['train'][:, :, 1] = 0.5

# Train env_1
# p(e=0) = 1
p_e['env_1'][:, :, 0] = 1

# Train env_2
# p(e=1) = 1
p_e['env_2'][:, :, 1] = 1

# Test Enviroment (env_3)
# p(e=2)=1
p_e['env_3'][:, :, 2] = 1

# Color

# p(c=0|y=0,e=0) =  p(c=1|y=1,e=0) = 0.9
p_c_ye[:, 0, 0, 0] = p_c_ye[:, 1, 0, 1] = 0.9
p_c_ye[:, 1, 0, 0] = p_c_ye[:, 0, 0, 1] = 0.1

# p(c=0|y=0,e=1) = p(c=1|y=1,e=1) = 0.8
p_c_ye[:, 0, 1, 0] = p_c_ye[:, 1, 1, 1] = 0.8
p_c_ye[:, 1, 1, 0] = p_c_ye[:, 0, 1, 1] = 0.2

# p(c=0|y=0,e=2) = p(c=1|y=1,e=2) = 0.1
p_c_ye[:, 0, 2, 0] = p_c_ye[:, 1, 2, 1] = 0.1
p_c_ye[:, 1, 2, 0] = p_c_ye[:, 0, 2, 1] = 0.9

# p(yec|de)
p_yc_de = p_y_d * p_c_ye

export_dir = '.'

for name, p_env in p_e.items():
    p_yec_d = p_yc_de * p_env
    torch.save(p_yec_d, os.path.join(export_dir, 'CMNIST_%s.pyt' % name))
