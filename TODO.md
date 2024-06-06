
# Loss
- [ ] gaze loss
- [ ] pairwise head pose and facial dynamics transfer loss
>  Let Ii and Ij be two frames randomly sampled from the same video of a subject. We extract their latent  variables using the encoders, and transfer Ii’s head pose onto Ij as ˆIj,zpose i = D(Vapp j , zid j , zpose i , zdyn j) and Ij’s facial motion onto Ii as Iˆi,zdyn j = D(Vapp i , zid i , zpose i , zdyn j ). The discrepancy loss between Iˆj,zpose i and Iˆi,zdyn j is subsequently minimized.
- [ ] face identity similarity loss

# Data

- [ ] High Res data


python -m torch.distributed.launch --nproc_per_node=2 train_dataset.py

ps aux|grep python|grep -v grep | awk '{print $2}'|xargs kill -9

nohup python -m torch.distributed.launch --nproc_per_node=8 train_dataset.py > log_files/240605_gpus8.log 2>&1 &