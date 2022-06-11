# python3
# Creat date: 2022-05-13
# Author: Chengchao.SUn
# Func: Dubbo -> 拉取ABtest的参数
# preference: 
#   1. python 调用 dubbo 接口: https://blog.csdn.net/sinat_38682860/article/details/108849575
#   2. Python3&Python2,通过zk,Telnet 调用dubbo: https://blog.csdn.net/wakuangyun/article/details/115675513
# =================================================================================================


import telnetlib
import time
from urllib.parse import unquote
from kazoo.client import KazooClient
import random
import threading, socket
from sys import platform, maxsize



class ServiceNotAvailableError(ValueError):
    pass


class Zookeeper:
    client = None
    service_dict = {}
    def __init__(self, host_port, timeout=120):
        self.host, self.port = host_port.split(":")
        self.host_port = host_port
        # 连接zookeeper
        self.client = KazooClient(host_port, timeout=timeout)
        self.client.start()

        # 查找所有注册的dubbo服务
        service_list = self.client.get_children('dubbo')
        for service in service_list:
            name = str(service).split('.')[-1]  # 去掉包名，剩下的服务名作为key
            self.service_dict[name] = service  # 此处如果有重名的服务，会覆盖

    def get_service_address(self, service):
        """获取指定服务的注册地址信息"""
        if '.' not in service:
            # 如果传入的服务名不带包名，就从service_dict找到完整服务名
            service = self.service_dict[service]

        uri = 'dubbo/%s' % service
        if not self.client.exists(uri):
            raise ServiceNotAvailableError('服务"%s"不存在' % service)
        elif not self.client.exists('%s/providers' % uri):
            raise ServiceNotAvailableError('服务"%s"没有提供者' % service)
        else:
            providers = self.client.get_children('%s/providers' % uri)
            addrs = []
            for provider in providers:
                addr = str(unquote(provider)).split('/')[2]
                addrs.append((str(addr).split(':')[0], str(addr).split(':')[1], str(addr)))
            return addrs

    def close(self):
        self.client.stop()
        self.client



class Dubbo:
    def __init__(self, interface, host_port, zkClient=None):
        self.host, self.port = host_port.split(":")
        self.conn = telnetlib.Telnet()
        self.interface = interface + '.'
        self.zk = zkClient
    
    def __call__(self, method, method_args_list):
        return self.request(method, method_args_list)


    def request(self, method, method_args_list):
        if self.zk is not None:
            self._get_zk_host_port()

        self.conn.open(self.host, self.port, timeout=120)
        cmd_ = self.generate_cmd(method, method_args_list)
        self.conn.write('invoke {}\n'.format(cmd_).encode())
        res = self.conn.read_until('dubbo>'.encode()).decode().split('\r\n')
        return res


    def generate_cmd(self, method, method_args_list):
        param_str = '('
        if len(method_args_list):
            for p in method_args_list:
                param_str += str(p)
        param_str += ')'
        cmd_ = self.interface + method + param_str
        return cmd_
    
    def _get_zk_host_port(self):
        address_list = self.zk.get_service_address(self.interface[:-1])
        if len(address_list) > 1:
            # 对于多节点服务，默认连接第一个节点，可用index指定
            print('——' * 43)
            print('|%s服务有多个地址，使用index参数指定请求地址，随机圈选：|' % str(self.interface).center(30, ' '))
            print('-' * 86)
            for i, address in enumerate(address_list):
                print('| %d ==> %s:%s |' % (i, address[0], str(address[1]).ljust(80 - len(address[2]), ' ')))
            print('——' * 43)

        self.host, self.port = random.choice(address_list)[:2]
        print('当前连接地址： %s:%s' % (self.host, self.port))

    def close(self):
        self.zk.close()





if __name__ == '__main__':
    serv_ = 'xx.xxxx.xx.xx.api.service.Xxxxx'
    zk = Zookeeper(host_dict['xx'])
    db_ = Dubbo(
        interface=serv_,
        host_port=host_dict['xx'], 
        zkClient=zk
    )
    res_ = db_.request(method='getLab', method_args_list=method_args)
    print("res_: \n", res_)
