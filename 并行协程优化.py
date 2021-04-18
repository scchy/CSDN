# python3
# Create Date: 2021-04-18
# Author: Scc_hy
# Func: 智慧海洋预处理优化实例
import pandas as pd
import numpy as np
import os 
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import time, datetime
import warnings 
warnings.filterwarnings(action='ignore')


def detect_total_memory(return_flag=True):
    mem_info = psutil.virtual_memory()
    mem_used = mem_info.used/ 1024 / 1024 / 1024
    mem_used_rate = mem_info.percent
    print(f'当前进程的内存使用情况：{mem_used_rate:.2f}% | {mem_used:.5f} GB')
    if return_flag:
        return mem_used, mem_used_rate

def detect_process_memory():
    now_p_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    print (f'当前进程的内存使用：{now_p_mem:.5f} GB | {now_p_mem * 1024:.3f} MB')


def clock(func):
    def clocked(*args, **kwargs):
        st = time.perf_counter()
        res = func(*args, **kwargs)
        cost = time.perf_counter() - st 
        print(f'{func.__name__} Done. | It costs {cost:.3f}s')
        return res
    return clocked


def zhhy_preprocess(df_in):
    """
    智慧海洋预处理, 将每个小文件进行处理
    """
    df = df_in
    if isinstance(df, str):
        df = pd.read_csv(df_in)

    df = df.iloc[::-1].reset_index(drop=True)
    # mem_mb = df.memory_usage().sum()/1024/1024
    # mem_mb*200
    # print(f'df used {mem_mb:.3f} MB')

    out_list = [df['渔船ID'].iloc[0]]
    df.drop('渔船ID', axis=1,  inplace=True)
    try:
        out_list.append(df.type.values[0])
        df.drop('type', axis=1,  inplace=True)
    except Exception as e:
        pass

    df['time'] = df['time'].apply(
        lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
    df['hour'] = df['time'].dt.hour
    df_diff = df.diff(1).iloc[1:]

    df_diff['time_seconds'] = df_diff['time'].dt.total_seconds()
    df_diff['dis'] = np.sqrt(df_diff['x']**2 + df_diff['y']**2)

    # 将速度限制一下速度 
    df['hour_day'] = 0
    df.loc[(df.hour > 8) & (df.hour < 18) , 'hour_day'] = 1
    ## 白天速度 
    df_tmp = df.loc[(df['速度'] > 1) & (df['速度'] < 11) & (df.hour_day == 1), :]
    std_3n = np.std(df_tmp['速度'].value_counts().index[:3])
    std_al = np.std(df_tmp['速度'].value_counts().index[:])
    if std_3n < std_al:
        sp_nd = round(np.mean(df_tmp['速度'].value_counts().index[:3]), 3)
    else:
        sp_nd = round(np.mean(df_tmp['速度'].value_counts().index[:2]), 3)

    ## 晚上
    df_tmp_n = df.loc[(df['速度'] > 1) & (df['速度'] < 11) & (df.hour_day == 0), :]
    std_3n = np.std(df_tmp_n['速度'].value_counts().index[:3])
    std_al = np.std(df_tmp_n['速度'].value_counts().index[:])
    if std_3n < std_al:
        sp_nd_n = round(np.mean(df_tmp_n['速度'].value_counts().index[:3]), 3)
    else:
        sp_nd_n = round(np.mean(df_tmp_n['速度'].value_counts().index[:2]), 3)

    df['x'] /= 50000.0                     # 获取经度值
    df['y'] /= 200000.0                     # 获取维度之维度值

    out_list.extend([
        df['time'].dt.day.nunique(),
        df['time'].dt.hour.min(),
        df['time'].dt.hour.max(),
        df['time'].dt.hour.value_counts().index[0],

        df['速度'].max(),
        df['速度'].mean(),

        df_diff['速度'].min(),
        df_diff['速度'].max(),
        df_diff['速度'].mean(),
        (df_diff['速度'] > 0).mean(),
        (df_diff['速度'] == 0).mean(), # 10

        df_diff['方向'].max(),
        df_diff['方向'].mean(),
        (df_diff['方向'] > 0).mean(),
        (df_diff['方向'] == 0).mean(),

        (df_diff['x'].abs() / df_diff['time_seconds']).min(),
        (df_diff['x'].abs() / df_diff['time_seconds']).max(), # 16
        (df_diff['x'].abs() / df_diff['time_seconds']).mean(),
        (df_diff['x'] > 0).mean(),
        (df_diff['x'] == 0).mean(),

        (df_diff['y'].abs() / df_diff['time_seconds']).max(),
        (df_diff['y'].abs() / df_diff['time_seconds']).mean(),
        (df_diff['y'] > 0).mean(),
        (df_diff['y'] == 0).mean(),

        df_diff['dis'].min(),
        df_diff['dis'].max(), # 25
        df_diff['dis'].mean(), # 26

        (df_diff['dis']/df_diff['time_seconds']).max(),
        (df_diff['dis']/df_diff['time_seconds']).mean(),
        ## 增加速度的前 3的均值  如果前三的标准差低于 总体的就取前三 否则取前2
        sp_nd,
        sp_nd_n
    ])

    return out_list


@clock
def parallel_deal(func_, params_list, run_flag=True):
    workers = os.cpu_count() * 9 // 10
    p_ = ProcessPoolExecutor(workers)
    tasks = [p_.submit(func_, parami) for parami in params_list]
    # 我们可以得知，在提交线程任务的时候就会将参数全部提交进入内存中
    print('--'*20)
    print('After submit to process')
    detect_process_memory()
    if run_flag:
        print('Start parellel process')
        res = [task.result() for task in as_completed(tasks)]
        return res
    return []



def parallel_deal_generator(func_, params_list, chunksize = None):
    workers = os.cpu_count() * 9 // 10
    if not chunksize:
        chunksize = workers
    print('chunksize:', chunksize)
    p_ = ProcessPoolExecutor(workers)
    tasks = []
    chunk_n = 1
    n = 1
    while True:
        try:
            tasks.append(p_.submit(func_, next(params_list)))
            n += 1
            if chunksize == n:
                print('chunk_n:', chunk_n)
                chunk_n += 1
                n = 1
                print('--'*20)
                print('After submit to process')
                detect_process_memory()
                print('Start parellel process')
                yield [task.result() for task in as_completed(tasks)]
                tasks = []
        except:
            if len(tasks) > 0:
                yield [task.result() for task in as_completed(tasks)]
            break


@clock
def res2df(res): 
    if isinstance(res, list):
        return pd.DataFrame(res)
    return pd.DataFrame(list(res))


@clock
def main(only_detect_preocess_memory_flag=False):
    print('start processing')
    detect_process_memory()
    detect_total_memory()
    train_data_root = r'D:\Python_data\My_python\Projects\智慧海洋\data\hy_round1_train_20200102'
    params_list = (pd.read_csv( os.path.join(train_data_root, i) ) for i in os.listdir(train_data_root)[:200])
    # params_list = (os.path.join(train_data_root, i) for i in os.listdir(train_data_root)[:200])
    run_ = not only_detect_preocess_memory_flag
    print('run_: ', run_)
    res = parallel_deal(zhhy_preprocess, params_list, run_flag=run_)
    if not only_detect_preocess_memory_flag:        
        print('--'*20)
        print('start transform to dataFrame')
        res_df = res2df(res)
        print('res_df simple view')
        print('res_df.shape: ', res_df.shape)
        print('res_df.head: \n', res_df.head(2))
    print('Done')


@clock
def main_g():
    print('start processing')
    detect_process_memory()
    detect_total_memory()
    train_data_root = r'D:\Python_data\My_python\Projects\智慧海洋\data\hy_round1_train_20200102'
    params_list = (os.path.join(train_data_root, i) for i in os.listdir(train_data_root)[:200])

    res_g = parallel_deal_generator(zhhy_preprocess, params_list)
    res_df_list = []
    while True:    
        try:
            print('--'*20)
            res_df = res2df(next(res_g))
            res_df_list.append(res_df)
            print('res_df.shape: ', res_df.shape)
        except:
            break

    res_fdf = pd.concat(res_df_list)
    print('res_fdf.shape: ', res_fdf.shape)
    print('Done')


if __name__ == '__main__':
    main(False)
    main_g()
