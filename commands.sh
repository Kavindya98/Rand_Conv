"python train_digits.py --net digit -ni 10000 -vi 250 -bs 32 -lr 0.0001 -sc mnist10k -chs 3 -rc -vwr -nv 1"

"python train_digits.py --net digit -ni 10000 -vi 250 -bs 32 -lr 0.0001 -sc mnist10k -chs 3 -rc -mix -vwr -nv 1"

"python train_digits.py --net Lenet -ni 10000 -vi 250 -bs 32 -lr 0.0001 -sc mnist10k -chs 3 --rp1 --rp1_out_channel 16"