mkdir -p trained_rbm
mkdir -p trained_dbn
mkdir -p out/dbn
mkdir -p out/rbm



rm trained_rbm/*
rm trained_dbn/*
rm out/rbm/viz_recon/*
rm out/rbm/viz_rf/*
rm out/rbm/viz_weights/*
rm out/rbm/weights/*
#python3 debug_dbn.py
python3 run.py