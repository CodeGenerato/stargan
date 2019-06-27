CUDA_VISIBLE_DEVICES=4 python GAN/main.py --mode train --image_dir data/FullDataset \
--sample_dir outputs/run6/samples --log_dir outputs/run6/logs \
--model_save_dir outputs/run6/models --test_iters 200000
