To reproduce the experiments and save the weights, install the requirements from the main repository and then run (runs on a single RTX 2080 GPU):

    For the OT-Conditional Flow Matching method:

python3 train_cifar10.py --model "otcfm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000

    For the Independent Conditional Flow Matching (I-CFM) method:

python3 train_cifar10.py --model "icfm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000

    For the original Flow Matching method:

python3 train_cifar10.py --model "fm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000

    For the Stochastic Interpolants (SI) method:

python3 train_cifar10.py --model "si" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000


── Amortised OT methods (Two-Phase Protocol) ─────────────────────────────────

    For RA-OT (Regression-Amortized OT, Phase 1: M=50 batches, L=100 projections):

python3 train_cifar10.py --model "ra-ot" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --pretrain_M 50 --pretrain_L 100 --pretrain_eps 0.1 --pretrain_ridge 1e-3

    For OA-OT (Objective-Amortized OT, Phase 1: M=50 batches, L=100 projections, T=5000 steps):

python3 train_cifar10.py --model "oa-ot" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --pretrain_M 50 --pretrain_L 100 --pretrain_eps 0.1 --pretrain_T 5000 --pretrain_lr 1e-3

Note: Timing reports (pretrain time, U-Net train time, total time) are automatically saved to
results/<model>/<model>_timing_report.txt after training completes.


To compute the FID at end of training, run:

python3 compute_fid.py --model "otcfm" --step 400000 --integration_method dopri5
python3 compute_fid.py --model "ra-ot"  --step 400000 --integration_method dopri5
python3 compute_fid.py --model "oa-ot"  --step 400000 --integration_method dopri5