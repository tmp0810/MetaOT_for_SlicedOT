# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import sys

import torch
from absl import flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")

FLAGS(sys.argv)


# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
#PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
#PATH = "/root/ft_ra-ot_cifar10_weights_step_240.pt"
#PATH = "/root/ft_otcfm_cifar10_weights_step_240.pt"
PATH = "/root/ft_icfm_cifar10_weights_step_240.pt"




print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()


class NFECounter:
    def __init__(self, model):
        self.model = model
        self._cur = 0
        self.batch_nfes = []

    def reset_batch(self):
        self._cur = 0

    def end_batch(self):
        self.batch_nfes.append(self._cur)

    def __call__(self, t, x):
        self._cur += 1
        return self.model(t, x)


nfe_model = NFECounter(new_net)


# Define the integration method if euler is used
if FLAGS.integration_method == "euler":
    node = NeuralODE(new_net, solver=FLAGS.integration_method)


def gen_1_img(unused_latent):
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, 2, device=device)
            nfe_model.reset_batch()
            traj = odeint(
                nfe_model,
                x,
                t_span,
                rtol=FLAGS.tol,
                atol=FLAGS.tol,
                method=FLAGS.integration_method,
            )
            nfe_model.end_batch()
    traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    return img


print("Start computing FID")
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_tensorflow",
)
print()
print("FID has been computed")
print()
print("FID: ", score)
print()
if FLAGS.integration_method == "euler":
    nfe_per_sample = float(FLAGS.integration_steps)
    print(f"NFE / sample: {nfe_per_sample:.0f}  (fixed-step Euler, steps={FLAGS.integration_steps})")
else:
    nfes = nfe_model.batch_nfes
    mean_nfe = sum(nfes) / len(nfes)
    print(f"NFE / sample: {mean_nfe:.2f}")
    print(f"  (avg over {len(nfes)} batches | min={min(nfes)} max={max(nfes)})")
