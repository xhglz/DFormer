from .DFormerv2_S_Lite_mlp import *

"""Student keeps the deployment-friendly S_Lite + MLP setting"""
C.backbone = "DFormerv2_S_Lite_mlp"
C.decoder = "MLPDecoder"
C.decoder_embed_dim = 256

"""Distill Settings"""
C.teacher_config = "local_configs.NYUDepthv2.DFormerv2_S"
C.teacher_weight = "checkpoints/trained/DFormerv2_Small_NYU.pth"  
C.student_weight = (
    "checkpoints/NYUDepthv2_DFormerv2_S_Lite_mlp/epoch-230_miou_51.7.pth"
)
C.kd_temperature = 4.0
C.kd_weight = 0.5
C.feat_weight = 0.15
C.feat_indices = (2, 3)
C.student_feat_channels = [64, 128, 256, 512]
C.teacher_feat_channels = [64, 128, 256, 512]

"""Distill Train Config"""
C.lr = 3e-5
C.batch_size = 8
C.nepochs = 100
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.warm_up_epoch = 2
C.drop_path_rate = 0.15
C.train_scale_array = [0.75, 1.0, 1.25]

"""Eval Config"""
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [384, 512]

"""Store Config"""
C.checkpoint_start_epoch = 60
C.checkpoint_step = 4

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_distill")
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
