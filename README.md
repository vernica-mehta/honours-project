# honours-project

## command line commands to run scripts

sfh.py

```shell
nohup python3 -u sfh.py <n> <sfh_len> --n_jobs 20 > output.log 2>&1 &
```

snr.py -- add noise to already generated synthetic spectra sets
```shell
python3 snr.py <file> <snr>
```

cannon-train.py -- to do k-fold cross-validation

```shell
nohup python3 cannon-train.py <file> --kfold <folds> > output.log 2>&1 &
```

trainingsettest.py -- train and test cannon model against test set of set size

```shell
nohup python3 trainingsettest.py <file> <size> > output.log 2>&1 &
```
