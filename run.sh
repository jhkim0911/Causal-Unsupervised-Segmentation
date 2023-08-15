dataset="cityscapes"
train_gpu="0,1,2,3"
test_gpu="0"
ckpt="checkpoint/dino_vit_small_8.pth"
port="12301"
# CNN
# python train_front_door.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python fine_tuning.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python test.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu

# DETR
python train_front_door2.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python fine_tuning2.py --dataset $dataset --ckpt $ckpt --gpu $train_gpu --port $port && python test2.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu

# only test
# python test.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu
# python test2.py --dataset $dataset --ckpt $ckpt --gpu $test_gpu