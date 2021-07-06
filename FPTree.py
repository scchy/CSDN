# python3
# Create date: 2021-07-06
# Author: Scc_hy
# Func: FP Tree
# preference: https://www.cnblogs.com/pinard/p/6307064.html
# =================================================================================

from collections import Counter
from treelib import Tree
import os
import numpy as np
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")


class FPTree():
    def __init__(self, least_support=1, parallel_flag=False):
        self.least_support = least_support
        self.parallel_flag = parallel_flag
    
    def supports_parallel(self, trans_dict):
        workers = os.cpu_count() * 4 // 5
        with ProcessPoolExecutor(workers) as executor:
            out_iter = executor.map(Counter, list(trans_dict.values()))
        return self._clip(reduce(lambda x, y: x+y, out_iter))

    def supports_loop(self, trans_dict):
        out = Counter()
        for i in trans_dict.values():
            out += Counter(i)
        return self._clip(out)
    
    def supports(self, data):
        return self.supports_parallel(data) if self.parallel_flag else self.supports_loop(data)
    
    def _clip(self, support_dict):
        out = Counter()
        for k, v in support_dict.items():
            if v < self.least_support:
                continue
            out[k] = v
        return out

    def creat_flush_node(self, tree, identifier, parent=None):
        try:
            tree.create_node(tag=identifier, identifier=identifier, parent=parent, data=1)
        except Exception:
            tree[identifier].data += 1
        return tree

    def creat_flushed_tree(self, trans_dict, supports_dict):
        tree_ = Tree()
        root_parent = 'None_x'
        tree_.create_node(tag=root_parent, identifier=root_parent, parent=None, data=1)
        for trans_value_i in trans_dict.values():
            sorted_values = sorted(trans_value_i, key=lambda x: supports_dict[x], reverse=True)
            n = 0
            for bucket_item in sorted_values:
                if bucket_item not in supports_dict:
                    continue
                if n == 0:
                    tree_ = self.creat_flush_node(tree_, f'None_{bucket_item}', root_parent)
                elif n == 1:
                    tree_ = self.creat_flush_node(tree_, f'{sorted_values[n-1]}_{bucket_item}', f'None_{sorted_values[n-1]}')
                else:
                    tree_ = self.creat_flush_node(
                        tree_,
                        f'{sorted_values[n-1]}_{bucket_item}',
                        f'{sorted_values[n-2]}_{sorted_values[n-1]}'
                    )
                n += 1
        return tree_
    
    def fit(self, data):
        supports_dict = self.supports(data)
        self.tree = self.creat_flushed_tree(data, supports_dict)
        self.tree.show()

    def test_data(self):
        trans_dict = {
            'T01': 'A B C E F O'.split(' '),
            'T02': 'A C G'.split(' '),
            'T03': 'E I'.split(' '),
            'T04': 'A C D E G'.split(' '),
            'T05': 'A C E G L'.split(' '),
            'T06': 'E J'.split(' '),
            'T07': 'A B C E F P'.split(' '),
            'T08': 'A C D'.split(' '),
            'T09': 'A C E G M'.split(' '),
            'T10': 'A C E G N'.split(' ')
        }
        return trans_dict


if __name__ == '__main__':
    fp = FPTree(2)
    dt = fp.test_data()
    sp_dict = fp.supports(dt)
    print(sp_dict)
    fp.fit(dt)


