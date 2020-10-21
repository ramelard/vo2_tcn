import itertools

filters = [2,4,8,16,24]
kernels = [1,2,4,8]
kernels = [3,5,6,7]
max_dilation_pow = [1,2,3,4,5]
dropout_rate = [0.2]
learning_rate = [0.0005]
note = "glost"


task_id = 1
with open('glost_task_list.txt','w') as file:
    for f, k, d, dr, lr in itertools.product(filters, kernels, max_dilation_pow, dropout_rate, learning_rate):
        task_str = f'f{f}_k{k}_d{d}_dr{dr}_lr{lr}'
        task = f'arg={task_id} && touch $SLURM_TMPDIR/gpuidx && echo "Using $SLURMD_NODENAME (nodeid $SLURM_NODEID)" > ~/scratch/log-${{SLURM_JOBID}}_${{arg}}.txt && ./train.sh {f} {k} {d} {dr} {lr} {note} ${{SLURM_JOBID}}/{task_str} &>> ~/scratch/log-${{SLURM_JOBID}}_${{arg}}.txt && echo ${{arg}},$(date +"%H:%M:%S") >> ~/scratch/runtime-${{SLURM_JOBID}}.txt' 
        file.write(task + '\n')
        task_id = task_id + 1
