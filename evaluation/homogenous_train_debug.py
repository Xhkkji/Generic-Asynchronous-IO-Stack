import argparse, datetime
import os
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset, GeneratedDGLDataset
import csv 
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS
from GIDS import GIDS_DGLDataLoader

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


def env_int(name, fallback):
    value = os.getenv(name)
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def parse_step_list(raw_value):
    if raw_value is None:
        return []

    steps = []
    for part in raw_value.split(','):
        part = part.strip()
        if not part:
            continue
        steps.append(int(part))
    return sorted(set(steps))


def configure_async_debug_env(args, step, last_signature):
    is_target_step = step in args.async_debug_step_set
    debug_rows = args.async_debug_rows if is_target_step else args.async_debug_default_rows
    debug_rows = max(debug_rows, 0)
    warp_ctx_sample = max(args.async_debug_warp_ctx_sample, 0)
    debug_dims = max(args.async_debug_dims, 0)

    os.environ["GIDS_ASYNC_DEBUG_ROWS"] = str(debug_rows)
    os.environ["GIDS_ASYNC_DEBUG_DIMS"] = str(debug_dims)
    os.environ["GIDS_WARP_CTX_DEBUG_SAMPLE"] = str(warp_ctx_sample)

    signature = (
        debug_rows,
        debug_dims,
        warp_ctx_sample,
        is_target_step,
    )
    current_enabled = (debug_rows > 0) or (warp_ctx_sample > 0)
    previous_enabled = False
    if last_signature is not None:
        previous_enabled = (last_signature[0] > 0) or (last_signature[2] > 0)

    if signature != last_signature and (current_enabled or previous_enabled):
        print(
            "[async-debug] next step {}: rows={} dims={} ctx_sample={} target={}".format(
                step,
                debug_rows,
                debug_dims,
                warp_ctx_sample,
                int(is_target_step),
            )
        )

    return signature

# 设置数据的根目录、DGL 数据集目录和 PyTorch Geometric 数据集目录
data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
dgl_root = os.path.join(data_root, 'dgl_datasets')
pyg_root = os.path.join(data_root, 'pyg_datasets')
user_root = os.path.join("/home/lzl/nfs.d/dataset/graph_embedding/LinkPrediction/train_data/")
rmat_root = os.path.join("/home/lzl/nfs.d/dataset/graph_embedding/graph_data/")
# 确保目录存在，如果不存在则创建
for path in [data_root, dgl_root, pyg_root]:
    os.makedirs(path, exist_ok=True)

# --- 新增：用户自定义数据集配置字典 ---
USER_DATASET_CONFIG = {
    'com': {
        'edgelist_path': os.path.join(user_root, 'com_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'LJ': {
        'edgelist_path': os.path.join(user_root, 'LJ_srt_wei_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 16,
    },
    'soc': {
        'edgelist_path': os.path.join(user_root, 'soc_srt_wei_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'wv': {
        'edgelist_path': os.path.join(user_root, 'wv_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'ytb': {
        'edgelist_path': os.path.join(user_root, 'ytb_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 100,
    },
    'uk': {
        'edgelist_path': os.path.join(rmat_root, 'uk2007_srt_weg.txt'),
        'feature_dim': 64, 'hidden_dim': 64, 'num_classes': 10,
    },
    'pa': {
        'edgelist_path': os.path.join(rmat_root, 'pa_srt_weg_commneg.txt'),
        'feature_dim': 64, 'hidden_dim': 64, 'num_classes': 10,
    },
    'twt': {
        'edgelist_path': os.path.join(user_root, 'twt.edge'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    }
}

@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)

def track_acc_GIDS(g, args, device, label_array=None):
    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,
        window_buffer = args.window_buffer,
        wb_size = args.wb_size,
        accumulator_flag = args.accumulator,
        cache_dim = args.cache_dim
    
    )
    dim = args.emb_size

    if(args.accumulator):
        GIDS_Loader.set_required_storage_access(args.bw, args.l_ssd, args.l_system, args.num_ssd, args.peak_percent)


    if(args.cpu_buffer):
        num_nodes = g.number_of_nodes()
        num_pinned_nodes = int(num_nodes * args.cpu_buffer_percent)
        GIDS_Loader.cpu_backing_buffer(dim, num_pinned_nodes)
        pr_ten = torch.load(args.pin_file)
        GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)


    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')]
               )

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    in_feats = g.ndata['features'].shape[1]

    train_dataloader = GIDS_DGLDataLoader(
        g,
        train_nid,
        sampler,
        args.batch_size,
        dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
        )

    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers, args.num_heads).to(device)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

    train_iter = iter(train_dataloader)
    train_iter.start_profiling("./gids_profile")
    print(f"start training...")
    warm_up_iter = 100
    # Setup is Done
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()
        epoch_loss = 0
        train_acc = 0
        model.train()

        batch_input_time = 0
        train_time = 0
        transfer_time = 0
        e2e_time = 0
        e2e_time_start = time.time()
        debug_signature = None
        debug_signature = configure_async_debug_env(args, 0, debug_signature)
        # for step, (input_nodes, seeds, blocks, ret) in tqdm.tqdm(enumerate(train_dataloader)):
        for step, (input_nodes, seeds, blocks, ret) in tqdm.tqdm(enumerate(train_iter)):    
            # print("step: ", step)
            
            if(step == warm_up_iter):
                print("warp up done")
                train_dataloader.print_stats()
                train_dataloader.print_timer()
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()

            if args.stop_after_step >= 0 and step >= args.stop_after_step:
                break
            debug_signature = configure_async_debug_env(args, step + 1, debug_signature)
        
            
            # Features are fetched by the baseline GIDS dataloader in ret

            batch_inputs = ret
            transfer_start = time.time() 

            batch_labels = blocks[-1].dstdata['labels']
            

            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
            transfer_time = transfer_time +  time.time()  - transfer_start
 
            # Model Training Stage
            train_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()                  
            train_time = train_time + time.time() - train_start
            
            if(step == warm_up_iter + 100):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start 
                train_dataloader.print_stats()
                train_dataloader.print_timer()
                print_times(transfer_time, train_time, e2e_time)
             
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                
    # 打印性能结果
    train_iter.stop_profiling()


    # Evaluation
    print(f'Evaluation:')
    model.eval()
    predictions = []
    labels = []
    # 自加######################################################
    # total_correct = 0
    # total_samples = 0
    # # 启用推理优化
    # torch.backends.cudnn.benchmark = True
    # 自加######################################################
    
    with torch.no_grad():
        for _, _, blocks in tqdm.tqdm(test_dataloader):
            # blocks = [block.to(device) for block in blocks]
            blocks = [block.int().to(device) for block in blocks]  # 改
            inputs = blocks[0].srcdata['feat']
     
            if(args.data == 'IGB' or args.data == 'CUSTOM'):
                labels.append(blocks[-1].dstdata['label'])
            elif(args.data == 'OGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                # out_label = torch.index_select(label_array, 0, b[1]).flatten()
                # labels.append(out_label.numpy())
            # predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predict = model(blocks, inputs).argmax(1)
            predictions.append(predict)
        
        # predictions = np.concatenate(predictions)
        # labels = np.concatenate(labels)
        predictions = np.concatenate([pred.cpu() for pred in predictions])
        labels = np.concatenate([label.cpu() for label in labels])
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100

        
        #     # 异步数据传输
        #     blocks = [block.to(device, non_blocking=True) for block in blocks]
        #     inputs = blocks[0].srcdata['feat']
            
        #     # 所有计算保持在GPU上
        #     predictions = model(blocks, inputs).argmax(1)
            
        #     # 获取标签（根据你的数据集调整）
        #     if args.data in ['IGB', 'OGB']:
        #         batch_labels = blocks[-1].dstdata['label']
        #     else:
        #         batch_labels = blocks[-1].dstdata['labels']  # 注意：训练用'labels'，测试用'label'
            
        #     # 在GPU上计算正确预测数
        #     correct = (predictions == batch_labels).sum().item()
        #     total_correct += correct
        #     total_samples += batch_labels.size(0)
        
        # test_acc = total_correct / total_samples * 100
        
        
    print("Test Acc {:.2f}%".format(test_acc))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 172, 16, 10], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    parser.add_argument('--custom_root', type=str, default=data_root,
        help='Root directory containing generated datasets (default: ../data)')
    parser.add_argument('--custom_dataset_name', type=str, default='com',
        help='Dataset name under custom_root')
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='25,10')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=256)  # 默认256
    parser.add_argument('--learning_rate', type=float, default=0.03)  # 0.01
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    parser.add_argument('--num_ssd', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)

    parser.add_argument('--device', type=int, default=0)

    #GIDS Optimization
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    parser.add_argument('--pin_file', type=str, default="/home/xhk/hyperion/GIDS/dataset/igb/pr_full.pt", 
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')



    #GPU Software Cache Parameters
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK
    parser.add_argument('--stop_after_step', type=int, default=-1,
        help='Inclusive training step limit for quick debug runs; set to -1 to disable')
    parser.add_argument('--async_debug_steps', type=str, default='',
        help='Comma-separated training steps that should use elevated async debug rows')
    parser.add_argument('--async_debug_rows', type=int, default=32,
        help='Async debug rows to use for the steps listed in --async_debug_steps')
    parser.add_argument('--async_debug_default_rows', type=int,
        default=env_int('GIDS_ASYNC_DEBUG_ROWS', 0),
        help='Async debug rows for non-target steps')
    parser.add_argument('--async_debug_dims', type=int, default=env_int('GIDS_ASYNC_DEBUG_DIMS', 8),
        help='Number of feature dimensions printed by async debug')
    parser.add_argument('--async_debug_warp_ctx_sample', type=int,
        default=env_int('GIDS_WARP_CTX_DEBUG_SAMPLE', 0),
        help='Warp ctx sample count for submit/wait debug dumps')


    args = parser.parse_args()
    args.async_debug_step_set = set(parse_step_list(args.async_debug_steps))
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)
    print("Stop After Step: ", args.stop_after_step)
    print("Async Debug Steps: ", sorted(args.async_debug_step_set))
    print("Async Debug Target Rows: ", args.async_debug_rows)
    print("Async Debug Default Rows: ", args.async_debug_default_rows)
    print("Async Debug Dims: ", args.async_debug_dims)
    print("Async Debug Warp Ctx Sample: ", args.async_debug_warp_ctx_sample)

    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "CUSTOM"):
        print("Dataset: CUSTOM")
        dataset = GeneratedDGLDataset(args)
        g = dataset[0]
        g = g.formats('csc')
    else:
        g=None
        dataset=None
    
    track_acc_GIDS(g, args, device, labels)
