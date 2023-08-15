dataset="cocostuff27"
train_gpu="4,5,6,7"
test_gpu="4"
ckpt="checkpoint/dino_vit_base_8.pth"
port="12302"
# CNN
# python train_front_door.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python fine_tuning.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python test.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu

# DETR
python train_front_door2.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python fine_tuning2.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python test2.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu

# only test
# python test.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu
# python test2.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu