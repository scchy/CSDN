# python3
# Create Date: 2023-04-18
# Func: llama try
# ========================================================



shell_op = """
# 安装git
yum install git
yum install -y bzip2
yum install dpkg
yum install md5

# 查看系统
dpkg --print-architecture
arch


# 安装anaconda https://repo.anaconda.com/archive/
wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-aarch64.sh
chmod +x Anaconda3-2023.03-Linux-aarch64.sh
./Anaconda3-2023.03-Linux-aarch64.sh


# path加入
cd ~
vi .bashrc
export PATH=/root/anaconda3/bin:$PATH
source ~/.bashrc

# 创建环境
conda create -n sccDeep --clone base
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 百度云下载数据
pip install bypy 
bypy info
bypy list
nohup bypy downfile LLaMA/7B/params.json > __th1.log &


内存不足还进行了升配
# CPU&内存: 4核(vCPU) 32 GiB  ~1.4元/时
# 操作系统: Alibaba Cloud Linux  3.2104 LTS 64位 ARM版 等保2.0三级版
# 实例规格: ecs.r8y.2xlarge 
---->
# CPU&内存: 8核(vCPU) 64 GiB  ~2.5元/时
# 操作系统: Alibaba Cloud Linux  3.2104 LTS 64位 ARM版 等保2.0三级版
# 实例规格: ecs.r8y.2xlarge 

"""

# 载入测试
import sys
import os
import torch
import torch.distributed as dist
import torch.distributed as dist
import json
from pathlib import Path
from rich.console import Console
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
sys.path.append('/root/llama/llama')
from llama import  ModelArgs, Transformer, Tokenizer, LLaMA
import  logging
from functools import wraps
from datetime import datetime
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
log.addHandler(handler)
log.info('start')
cs = Console()


def clock(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        st = datetime.now()
        res = func(*args, **kwargs)
        cost_ = datetime.now() - st
        func_name = func.__name__
        print(f'{func_name} Done ! It cost {cost_}.')
        return res
    return clocked



def load_7B_model():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # gloo cpu ; nccl - gpu
    dist.init_process_group('gloo', init_method='file:///tmp/tmpInit19', rank=local_rank, world_size=world_size)
    # torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)

    max_seq_len = 512
    max_batch_size = 2
    ckpt_dir = '/root/llama/llama/model/7B/'
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer_path = '/root/llama/llama/model/tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    log.info('load Transformer Struct')

    ckpt_path = os.path.join(ckpt_dir, 'consolidated.00.pth')
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    log.info('start load params')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    generator = LLaMA(model, tokenizer)
    return generator

# dmesg | egrep -i -B100 'killed process'
# Killed process 63907 (python) total-vm:40772484kB, anon-rss:30914228kB, file-rss:4kB, shmem-rss:0kB, UID:0 pgtables:61332kB oom_score_adj:
# 进行升配
generator = load_7B_model()

@clock
def ask_llama(pp):
    return generator.generate(
    [pp], max_gen_len=256, temperature=0.8, top_p=0.95
)



ask_llama('I believe the meaning of life is')

str_ = """
>>> ask_llama('I believe the meaning of life is')

["I believe the meaning of life is to help others and I intend to keep doing so.\nI would love to continue this by setting up a community that does fundraising and raises money to help people in need.\nA situation that really inspired me to start this was when I had to go to hospital and I was so grateful for the care that I got. If it hadn't been for the hospital I would have been in a very bad place and without them I would not be here now.\nI went to hospital because I got appendicitis and needed to have an operation. When I woke up I was so grateful for the care that I got. The nurses, doctors and everyone who helped me were so kind and caring. After being in hospital for 12 days I was told that the operation I had was a very rare one and it was done by a very small group of people in the UK. I had no idea that this surgery had been done in my hospital by such a small group of people. The nurses, doctors and everyone else who helped me were so kind and caring and if I hadn't had the operation I would not be here now.\nI want to set up a community that does fundraising"]

>>> ask_llama('I would like to travel to Japan. Help me draft a travel plan')
ask_llama Done ! It cost 0:03:11.220175.
["I would like to travel to Japan. Help me draft a travel plan. I am a diabetic. I am also allergic to bee stings.\nOk, I will try my best to help you. I am a Japanese living in the United States.\nSo I have to tell you that you will have a hard time in Japan.\nI will try to tell you some information about traveling in Japan.\nFirst of all, you need a visa if you want to stay longer than 90 days.\nIf you have a health insurance, you can get a 90 days visa.\n(http://www.jnto.go.jp/eng/indepth/health_insurance.html)\nThis is the website of the embassy.\nThere is also information about traveling in Japan on the website.\nI suggest you contact them to find out more.\nNow, I am going to tell you some information.\nIf you have a physical disability, you can apply for a free train pass.\nIt will be 100% free.\nI'm sorry, but I don't know about the train pass for diabetic.\nBut, if you take care of your diabetes, I think"]


"""

