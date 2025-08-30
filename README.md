# honours-project

## command line commands to run scripts

sfh.py
```shell
nohup python3 -u sfh.py <n> <sfh_len> --n_jobs 20 > output.log 2>&1 &
```

cannon-train.py -- to do k-fold cross-validation
```shell
nohup python3 cannon-train.py <file> --labels <labels> --kfold <folds> > output.log 2>&1 &
```

cannon-train.py -- for a simple training and test set
```shell
nohup python3 cannon-train.py <file> --labela <labels> > output.log 2>&1 &
```