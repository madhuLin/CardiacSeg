import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 也保險地把父目錄丟進去（你原本的寫法）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from functools import partial

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.air import session

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    Orientationd,
    ToNumpyd,
)
from monailabel.transform.post import Restored

from expers.args import get_parser, map_args_transform, map_args_optim, map_args_lrschedule, map_args_network
from data_utils.dataset import DataLoader, get_label_names, get_infer_data
from data_utils.data_loader_utils import load_data_dict_json
from data_utils.utils import get_pids_by_loader, get_pids_by_data_dicts
from runners.tuner import run_training
from runners.tester import run_testing
from runners.inferer import run_infering
from networks.network import network
from optimizers.optimizer import Optimizer, LR_Scheduler
import json

def main(config, args=None):
    if args.tune_mode == 'transform':
        args = map_args_transform(config, args)
    elif args.tune_mode == 'optim':
        args = map_args_optim(config['optim'], args)
    elif args.tune_mode == 'lrschedule' or args.tune_mode == 'lrschedule_epoch':
        args = map_args_transform(config['transform'], args)
        args = map_args_optim(config['optim'], args)
        args = map_args_lrschedule(config['lrschedule'], args)
    elif args.tune_mode == 'network':
        args = map_args_network(config, args)
    else:
        # for LinearWarmupCosineAnnealingLR
        args.max_epochs = args.max_epoch
        print('a_max', args.a_max)
        print('a_min', args.a_min)
        print('space_x', args.space_x)
        print('roi_x', args.roi_x)
        print('lr', args.lr)
        print('weight_decay', args.weight_decay)
        print('warmup_epochs',args.warmup_epochs)
        print('max_epochs',args.max_epochs)
    
    
    # train
    args.test_mode = False
    args.checkpoint = os.path.join(args.model_dir, 'final_model.pth')
    main_worker(args)
    # test
    args.test_mode = True
    args.checkpoint = os.path.join(args.model_dir, 'best_model.pth')
    args.ssl_checkpoint = None
    main_worker(args)
    


def main_worker(args):
    # # make dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    # model
    model = network(args.model_name, args)
    
    # loss
    if args.loss == 'dice_focal_loss':
        print('loss: dice focal loss')
        dice_loss = DiceFocalLoss(
            to_onehot_y=True, 
            softmax=True,
            gamma=2.0,
            lambda_dice=args.lambda_dice,
            lambda_focal=args.lambda_focal
        )
    else:
        print('loss: dice ce loss')
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        # print('loss: dice loss')
        # dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        # print('loss: dice focal loss')
        # dice_loss = DiceFocalLoss(
        #     to_onehot_y=True, 
        #     softmax=True,
        #     gamma=2.0,
        #     lambda_dice=args.lambda_dice,
        #     lambda_focal=args.lambda_focal
        # )
    
    # optimizer
    print(f'optimzer: {args.optim}')
    optimizer = Optimizer(args.optim, model.parameters(), args)

    # lrschedule
    if args.lrschedule is not None:
        print(f'lrschedule: {args.lrschedule}')
        scheduler = LR_Scheduler(args.lrschedule, optimizer, args)
    else:
        scheduler = None

    # check point
    start_epoch = args.start_epoch
    early_stop_count = args.early_stop_count
    best_acc = 0
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load lrschedule
        if args.lrschedule is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        # load check point epoch and best acc
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "early_stop_count" in checkpoint:
            early_stop_count = checkpoint["early_stop_count"]
        print(
          "=> loaded checkpoint '{}' (epoch {}) (bestacc {}) (early stop count {})"\
          .format(args.checkpoint, start_epoch, best_acc, early_stop_count)
        )
    else:
        # ssl pretrain
        if args.model_name =='swinunetr' and args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
            pre_train_path = os.path.join(args.ssl_checkpoint)
            weight = torch.load(pre_train_path)
           
            if "net" in list(weight["state_dict"].keys())[0]:
                print("Tag 'net' found in state dict - fixing!")
                for key in list(weight["state_dict"].keys()):
                    if 'swinViT' in key:
                        new_key = key.replace("net.swinViT", "module")
                        weight["state_dict"][new_key] = weight["state_dict"].pop(key) 
                    else:
                        new_key = key.replace("net", "module")
                        weight["state_dict"][new_key] = weight["state_dict"].pop(key)

                    if 'linear' in  new_key:
                        weight["state_dict"][new_key.replace("linear", "fc")] = weight["state_dict"].pop(new_key)
            
            model.load_from(weights=weight)
            print("Using pretrained self-supervied Swin UNETR backbone weights !")
            print(
              "=> loaded pretrain checkpoint '{}'"\
              .format(args.ssl_checkpoint)
            )
        elif args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
            model_dict = torch.load(args.ssl_checkpoint)
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            if "net" in list(state_dict.keys())[0]:
                print("Tag 'net' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("net.", "")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print(f"Using pretrained self-supervied {args.model_name} backbone weights !")
            print(
              "=> loaded pretrain checkpoint '{}'"\
              .format(args.ssl_checkpoint)
            )
            
    # inferer
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    # writer
    writer = SummaryWriter(log_dir=args.log_dir)

    if not args.test_mode:
        # load train and test data
        loader = DataLoader(args.data_name, args)()
        
        tr_loader, val_loader = loader
        
        # training
        run_training(
            start_epoch=start_epoch,
            best_acc=best_acc,
            early_stop_count=early_stop_count,
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=dice_loss,
            acc_func=dice_acc,
            model_inferer=model_inferer,
            post_label=post_label,
            post_pred=post_pred,
            writer=writer,
            args=args,
        )
    
    else:
        os.makedirs(args.eval_dir, exist_ok=True)
        
        
        label_names = get_label_names(args.data_name)
        
        # prepare data_dict
        _, _, test_dicts = load_data_dict_json(args.data_dir, args.data_dicts_json)
        
        # infer post transform
        keys = ['pred']
        if args.data_name == 'mmwhs' or args.data_name == 'mmwhs2':
            axcodes='LAS'
        else:
            axcodes='LPS'
        post_transform = Compose([
            Orientationd(keys=keys, axcodes=axcodes),
            ToNumpyd(keys=keys),
            Restored(keys=keys, ref_image="image")
        ])
       
        # run infer
        pids = get_pids_by_data_dicts(test_dicts)
        inf_dc_vals = []
        inf_iou_vals = []
        inf_sensitivity_vals = []
        inf_specificity_vals = []
        tt_dc_vals = []
        tt_iou_vals = []
        inf_times = []
        for data_dict in test_dicts:
            print('infer data:', data_dict)
            # load infer data
            data = get_infer_data(data_dict, args)
            # infer
            ret_dict = run_infering(
                model,
                data,
                model_inferer,
                post_transform,
                args
            )
            tt_dc_vals.append(ret_dict['tta_dc'])
            tt_iou_vals.append(ret_dict['tta_iou'])
            inf_dc_vals.append(ret_dict['ori_dc'])
            inf_iou_vals.append(ret_dict['ori_iou'])
            inf_sensitivity_vals.append(ret_dict['ori_sensitivity'])
            inf_specificity_vals.append(ret_dict['ori_specificity'])
            inf_times.append(ret_dict['inf_time'])
            

        
        # make df
        eval_tt_dice_val_df = pd.DataFrame(
            tt_dc_vals,
            columns=[f'tt_dice{n}' for n in label_names]
        )
        eval_tt_iou_val_df = pd.DataFrame(
            tt_iou_vals,
            columns=[f'tt_iou{n}' for n in label_names]
        )
        
        
        eval_inf_dice_val_df = pd.DataFrame(
            inf_dc_vals,
            columns=[f'inf_dice{n}' for n in label_names]
        )
        eval_inf_iou_val_df = pd.DataFrame(
            inf_iou_vals,
            columns=[f'inf_iou{n}' for n in label_names]
        )
        eval_inf_sensitivity_val_df = pd.DataFrame(
            inf_sensitivity_vals,
            columns=[f'inf_sensitivity{n}' for n in label_names]
        )
        eval_inf_specificity_val_df = pd.DataFrame(
            inf_specificity_vals,
            columns=[f'inf_specificity{n}' for n in label_names]
        )
        
        
        eval_inf_time_df = pd.DataFrame(
            inf_times,
            columns=[f'inf_time']
        )
        
        pid_df = pd.DataFrame({
            'patientId': pids,
        })
        
        avg_tt_dice = eval_tt_dice_val_df.T.mean().mean()
        avg_tt_iou =  eval_tt_iou_val_df.T.mean().mean()
        avg_inf_dice = eval_inf_dice_val_df.T.mean().mean()
        avg_inf_iou =  eval_inf_iou_val_df.T.mean().mean()
        avg_inf_sensitivity =  eval_inf_sensitivity_val_df.T.mean().mean()
        avg_inf_specificity =  eval_inf_specificity_val_df.T.mean().mean()
        avg_inf_time = eval_inf_time_df.T.mean().mean()

        eval_df = pd.concat([
            pid_df, eval_tt_dice_val_df, eval_tt_iou_val_df,
            eval_inf_dice_val_df, eval_inf_iou_val_df,
            eval_inf_sensitivity_val_df, eval_inf_specificity_val_df,eval_inf_time_df
        ], axis=1, join='inner').reset_index(drop=True)
        
        if args.save_eval_csv:
            eval_df.to_csv(os.path.join(args.eval_dir, f'best_model.csv'), index=False)
        
        print("\neval result:")
        print('avg tt dice:', avg_tt_dice)
        print('avg tt iou:', avg_tt_iou)
        print('avg inf dice:', avg_inf_dice)
        print('avg inf iou:', avg_inf_iou)
        print('avg inf sensitivity:', avg_inf_sensitivity)
        print('avg inf specificity:', avg_inf_specificity)
        print('avg inf time:', avg_inf_time)
        
        print(eval_df.to_string())
        
        # tune.report(
        #     tt_dice=avg_tt_dice,
        #     tt_iou=avg_tt_iou,
        #     inf_dice=avg_inf_dice,
        #     inf_iou=avg_inf_iou,
        #     val_bst_acc=best_acc,
        #     inf_time=avg_inf_time
        # )
        session.report({
            "tt_dice": float(avg_tt_dice),
            "tt_iou": float(avg_tt_iou),
            "inf_dice": float(avg_inf_dice),
            "inf_iou": float(avg_inf_iou),
            "val_bst_acc": float(best_acc) if isinstance(best_acc, (int, float)) else 0.0,
            "inf_time": float(avg_inf_time),
        })

def _resolve_best_model_path(best_result, args):
    """
    依序尋找可用權重：
    1) 明確指定的 --checkpoint
    2) Ray AIR Result.checkpoint（含 .path 或 .to_directory()）
    3) Ray AIR Result.path / 舊版 .log_dir 下的 models 目錄
    4) args.model_dir 下的 best_model.pth / final_model.pth
    """
    # 1) explicit --checkpoint
    ckpt_arg = getattr(args, "checkpoint", None)
    if ckpt_arg and os.path.isfile(ckpt_arg):
        return ckpt_arg

    # 2) AIR checkpoint object
    ckpt_obj = getattr(best_result, "checkpoint", None)
    if ckpt_obj:
        # 2a) checkpoint.path 直接有檔案夾
        ckpt_path = getattr(ckpt_obj, "path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            for name in ("best_model.pth", "final_model.pth", "checkpoint.pt", "checkpoint.pth"):
                p = os.path.join(ckpt_path, name)
                if os.path.isfile(p):
                    return p
        # 2b) 將 checkpoint 展開到臨時資料夾
        try:
            tmpdir = ckpt_obj.to_directory()
            for name in ("best_model.pth", "final_model.pth", "checkpoint.pt", "checkpoint.pth"):
                p = os.path.join(tmpdir, name)
                if os.path.isfile(p):
                    return p
        except Exception:
            pass

    # 3) Result 的路徑（新版用 result.path；舊版可能有 log_dir）
    logdir = getattr(best_result, "path", None) or getattr(best_result, "log_dir", None)
    if logdir and os.path.isdir(logdir):
        for name in ("best_model.pth", "final_model.pth"):
            p = os.path.join(logdir, "models", name)
            if os.path.isfile(p):
                return p

    # 4) 最後回退到 args.model_dir
    model_dir = getattr(args, "model_dir", None) or "./models"
    for name in ("best_model.pth", "final_model.pth"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "找不到可用的模型權重。請確認：\n"
        f"  - --checkpoint 是否存在: {ckpt_arg}\n"
        f"  - Result.checkpoint 是否包含檔案\n"
        f"  - {model_dir}/best_model.pth 或 final_model.pth 是否存在"
    )

def _result_dir(best_result, default_dir):
    # 依序嘗試新版/舊版欄位
    for attr in ("path", "log_dir"):
        p = getattr(best_result, attr, None)
        if p and os.path.isdir(p):
            return p
    return default_dir

if __name__ == "__main__":
    args = get_parser(sys.argv[1:])
    cardiac_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # ray.init(runtime_env={"working_dir": cardiac_root})
    ray.init(runtime_env={"excludes": [".git", "**/*.zip", "**/*.tar", "**/*.tar.gz"]})
    if args.tune_mode == 'test':
        print('test mode')
    elif args.tune_mode == 'train':
        search_space = {
            "exp": tune.grid_search([
              {
                  'exp': args.exp_name,
              }
            ])
        }
    elif args.tune_mode == 'network':
        search_space = {
            'depths': tune.grid_search([
                [4, 4, 12, 4],
            ])
        }
    elif args.tune_mode == 'transform':
        search_space = {
            'intensity': tune.grid_search([
                [-42, 423], 
            ]),
            'space': tune.grid_search([
                [0.4,0.4,0.5],
                [0.8,0.8,0.8],
                [0.8,0.8,1.0],
                [1.0,1.0,1.0],
            ]),
            'roi': tune.grid_search([
                [128,128,128],
            ]),
        }
    
    # {'lr': 5e-04, 'weight_decay': 5e-04},
    # {'lr': 7e-04, 'weight_decay': 5e-04},
    # {'lr': 9e-04, 'weight_decay': 5e-04},
    # {'lr': 7e-04, 'weight_decay': 3e-04},
    # {'lr': 7e-04, 'weight_decay': 7e-04},
    # {'lr': 7e-04, 'weight_decay': 9e-04},
    elif args.tune_mode == 'optim':
        search_space = {
            'optim': tune.grid_search([
                {'lr': 5e-04, 'weight_decay': 5e-04},
                {'lr': 7e-04, 'weight_decay': 5e-04},
                {'lr': 9e-04, 'weight_decay': 5e-04},
            ])
        }
    elif args.tune_mode == 'lrschedule':
        search_space = {
            'transform': tune.grid_search([
                {
                    'intensity': [-42,423],
                    'space': [0.7,0.7,1.0],
                    'roi':[128,128,128],
                }
            ]),
            'optim': tune.grid_search([
                {'lr':5e-5, 'weight_decay': 5e-4},
                {'lr':5e-4, 'weight_decay': 5e-5},
                {'lr':5e-4, 'weight_decay': 5e-3},
            ]),
            'lrschedule': tune.grid_search([
                {'warmup_epochs':60,'max_epoch':1200},
            ])
        }
    elif args.tune_mode == 'lrschedule_epoch':
        search_space = {
            'transform': tune.grid_search([
                {
                    'intensity': [-42,423],
                    'space': [1.0,1.0,1.0],
                    'roi':[128,128,128],
                }
            ]),
            'optim': tune.grid_search([
                {'lr':1e-2, 'weight_decay': 3e-5},
                {'lr':5e-3, 'weight_decay': 5e-4},
                {'lr':5e-4, 'weight_decay': 5e-5},
            ]),
            'lrschedule': tune.grid_search([
                {'warmup_epochs':40,'max_epoch':700},
                {'warmup_epochs':60,'max_epoch':700},
            ])
        }
    else:
        raise ValueError(f"Invalid args tune mode:{args.tune_mode}")

    trainable_with_cpu_gpu = tune.with_resources(partial(main, args=args), {"cpu": 1, "gpu": 1})
    reporter = CLIReporter(metric_columns=[
        'tt_dice',
        'tt_iou',
        'inf_dice',
        'inf_iou',
        'val_bst_acc',
        'inf_time',
    ])


    if args.resume_tuner:
        print(f'resume tuner form {args.root_exp_dir}')
        restored_tuner = tune.Tuner.restore(os.path.join(args.root_exp_dir, args.exp_name), trainable=trainable_with_cpu_gpu)
        
        # for manual test
        if args.tune_mode == 'test':
            print('run test mode ...')
            # get best model path
            result_grid = restored_tuner.get_results()
            best_result = result_grid.get_best_result(metric="inf_dice", mode="max")
            # model_pth = os.path.join( best_result.log_dir, 'models', 'best_model.pth')
            model_pth = _resolve_best_model_path(best_result, args)
            print(f"[INFO] 使用的測試權重：{model_pth}")
            # test
            # for LinearWarmupCosineAnnealingLR
            args.max_epochs = args.max_epoch
            args.test_mode = True
            args.checkpoint = os.path.join(model_pth)
            result_dir = _result_dir(best_result, args.root_exp_dir if hasattr(args, "root_exp_dir") else args.eval_dir)
            # 保持你 CLI 指定的 eval_dir 優先；若沒指定就用 result_dir/evals
            if not getattr(args, "eval_dir", None):
                args.eval_dir = os.path.join(result_dir, "evals")
            os.makedirs(args.eval_dir, exist_ok=True)
            print(f"[INFO] 評估輸出目錄：{args.eval_dir}")
            
            main_worker(args)
        else:
            result = restored_tuner.fit()
    else:
        tuner = tune.Tuner(
            trainable_with_cpu_gpu,
            param_space=search_space,
            run_config=air.RunConfig(
                name=args.exp_name,
                storage_path=str(args.root_exp_dir),
                progress_reporter=reporter
            )
        )
        tuner.fit()
    
