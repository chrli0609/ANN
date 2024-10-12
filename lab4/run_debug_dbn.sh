mkdir -p trained_rbm
mkdir -p trained_dbn
mkdir -p out/dbn
mkdir -p out/rbm



rm trained_rbm/*
python3 debug_dbn.py
