
echo "Result on GDSSR"
python inference_GDSSR.py -opt options/GDSSR_GAN_x4.yml --tile 200 \
    --im_path imgs \
    --model_path experiments/GDSSR_GAN.pth \
    --res_path results-GAN
