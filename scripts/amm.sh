YAMLS="--model configs/AMM.yaml --trainer configs/trainer.yaml "

# If you specified the parameters in the YAML file, the following line is enough
TRAINER="--trainer.max_epochs 10 "
MODEL=""

# Instead of updating YAML files, you can also specify the parameters directly
# TRAINER="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 10 " 
# MODEL="--model.dataset imagenet --model.data_dir /PATH/TO/IMAGENET --model.model resnet18 --model.batch_size_train 64 --model.learning_rate 1e-5 "

CUDA_VISIBLE_DEVICES=0 python -m src.main $YAMLS $TRAINER $MODEL \
 --model.f_loss location --model.h_lambda 10 --model.h_target_layer permute \
 --model.h_method grad-cam --model.model swin_v2_t \
 --trainer.val_check_interval 2000  --model.freeze_bn True

CUDA_VISIBLE_DEVICES=0 python -m src.main $YAMLS $TRAINER $MODEL \
 --model.f_loss location --model.h_lambda 10 --model.h_target_layer layer4 \
 --model.h_method grad-cam --model.model resnet18 \
 --trainer.val_check_interval 2000  --model.freeze_bn True


