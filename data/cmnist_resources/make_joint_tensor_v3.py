import torch
import os

# Conditional probability distributions are represented as tensors
p_e = {}
# Digit, Label, Environment, Color
p_d_e = torch.zeros([2, 2, 3, 2])
p_y_d = torch.zeros([2, 2, 3, 2])
p_e['train'] = torch.zeros([2, 2, 3, 2])
p_e['test'] = torch.zeros([2, 2, 3, 2])
p_c_ye = torch.zeros([2, 2, 3, 2])


# Label given environment

# p(d=0|e=0) = 0.6
p_d_e[0,:,0,:] = 0.6
p_d_e[1,:,0,:] = 0.4

# p(d=0|e=1) = 0.2
p_d_e[0,:,1,:] = 0.2
p_d_e[1,:,1,:] = 0.8

# p(d=0|e=2) = p(d=1|e=2) = 0.5
p_d_e[0,:,2,:] = p_d_e[1,:,2,:] = 0.5

# p(y=0|d=0) = p(y=1|d=1) = 0.75
p_y_d[0,0] = p_y_d[1,1] = 0.75
p_y_d[0,1] = p_y_d[1,0] = 0.25

# Train

# p(e=0|t=0) = 0.75
p_e['t0'][:, :, 0] = 0.75
# p(e=1|t=0) = 0.25
p_e['t0'][:, :, 1] = 0.25

# p(e=2|t=1)=1
p_e['t1'][:, :, 2] = 1

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

# p(ycd|e) = p(d|y)p(y|e)p(c|ey)
p_ycd_e = p_d_e * p_y_d * p_c_ye

export_dir = '.'

for name, p_env in p_e.items():
    # p(ycde) = p(ycd|e) * p(e)
    p_ycde = p_ycd_e * p_env

    # p(d) = \sum_{y,c,e} p(ycde)
    p_d = p_ycde.sum(3).sum(2).sum(1)

    # p(yec|d) = p(ycde)/p(d)
    p_yec_d = p_ycde / p_d.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    torch.save(p_yec_d, os.path.join(export_dir, 'CMNIST_v3_%s.pyt' % name))