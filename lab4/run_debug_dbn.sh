mkdir -p trained_rbm
mkdir -p trained_dbn
mkdir -p out/dbn
mkdir -p out/rbm



rm trained_rbm/*
rm trained_dbn/*
rm out/rbm/loss/*
rm out/rbm/viz_recon/*
rm out/rbm/viz_rf/*
rm out/rbm/viz_weights/*
rm out/rbm/weights/*
rm out/dbn/generate/*
rm out/dbn/label_values/*
rm out/dbn/recon_loss/*

#python3 debug_dbn.py
python3 run.py