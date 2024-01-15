cora
python train.py --dname 'cora' --num_layers 32 --cuda 0 --feature_noise 0.0 --lamda 0.12 --hidden 1024 --wd 0.0 --epochs 5000 --runs 10 --lalpha --lr 0.0015 --alpha 0.28 --dropout 0.7 --method 'GCNII' --nh --step_size 180 --gamma 0.05

cora_ca
python train.py --dname 'coauthor_cora' --num_layers 32 --cuda 1 --feature_noise 0.0 --lamda 0.5 --hidden 512 --wd 0.0 --epochs 5000 --runs 10 --lalpha --lr 0.01 --alpha 0.4 --dropout 0.7 --method 'GCNII' --nh --step_size 20 --gamma 0.5

dblp_ca
python train.py --dname 'coauthor_dblp' --num_layers 32 --cuda 5 --feature_noise 0.0 --lamda 0.5 --hidden 256 --wd 0.0 --epochs 5000 --runs 10 --lalpha --lr 0.01 --alpha 0.2 --dropout 0.5 --method 'GCNII' --nh

citeseer
python train.py --dname 'citeseer' --num_layers 64 --cuda 7 --feature_noise 0.0 --lamda 0.94 --hidden 256 --wd 0 --epochs 5000 --runs 10 --lalpha --lr 0.0019 --alpha 0.5 --dropout 0.6 --method 'GCNII' --nh --step_size 35 --gamma 0.5

pubmed
python train.py --dname 'pubmed' --num_layers 2 --cuda 7 --feature_noise 0.0 --lamda 0.1 --hidden 64 --wd 0 --epochs 5000 --runs 10 --lalpha --lr 0.008 --alpha 0.42 --dropout 0.5 --method 'GCNII' --nh --step_size 590 --gamma 0.5

congress
python train.py --dname 'congress-bills-100' --num_layers 1 --cuda 4 --feature_noise 1.0 --lamda 0.3 --hidden 512 --wd 0.0 --epochs 5000 --runs 10 --lalpha --lr 0.0005 --alpha 0.1 --dropout 0.2 --method 'GCNII' --nh --step_size 290 --gamma 0.9

senate
python train.py --dname 'senate-committees-100' --num_layers 2 --cuda 3 --feature_noise 1.0 --lamda 1.5 --hidden 512 --wd 0 --epochs 5000 --runs 10 --lalpha --lr 0.01 --alpha 0.2 --dropout 0.8 --method 'GCNII' --nh

walmart
python train.py --dname 'walmart-trips-100' --num_layers 4 --cuda 1 --feature_noise 1.0 --lamda 0.5 --hidden 256 --wd 0 --epochs 5000 --runs 10 --lalpha --lr 0.004 --alpha 0.1 --dropout 0.3 --method 'GCNII' --nh

house
python train.py --dname 'house-committees-100' --num_layers 4 --cuda 6 --feature_noise 1.0 --lamda 0.6 --hidden 64 --wd 0 --epochs 5000 --runs 10 --lalpha --lr 0.009 --alpha 0.2 --dropout 0.4 --method 'GCNII' --nh --step_size 30 --gamma 0.07


