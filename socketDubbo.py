# python3
# Creat date: 2022-06-11
# Author: Chengchao.SUn
# Func: Dubbo -> 拉取ABtest的参数
# preference: 
#   1. socket()函数用法详解：创建套接字  http://c.biancheng.net/view/2131.html
#   2. https://github.com/BSTester/dubbo-python
# =================================================================================================

import time
from urllib.parse import unquote
from kazoo.client import KazooClient
import random
import threading, socket, struct
from sys import platform, maxsize
from struct import unpack
import json
from datetime import datetime



response_status_message = {
    20: 'OK',
    30: 'CLIENT_TIMEOUT',
    31: 'SERVER_TIMEOUT',
    40: 'BAD_REQUEST',
    50: 'BAD_RESPONSE',
    60: 'SERVICE_NOT_FOUND',
    70: 'SERVICE_ERROR',
    80: 'SERVER_ERROR',
    90: 'CLIENT_ERROR'
}



def num_2_byte_list(num):
    """
    convert num to byte list
    :param num:
    :return:
    """
    byte = []
    while num > 0:
        b = num & 0xff  # 获取最低位的一个字节的值
        byte.append(b)
        num = num >> 8  # 移除最低位的一个字节
    return list(reversed(byte))



def get_request_body_length(body):
    """
    获取body的长度，并将其转为长度为4个字节的字节数组
    :param body:
    :return:
    """
    return list(bytearray(struct.pack('!i', len(body))))




def double_to_long_bits(value):
    """
    https://gist.github.com/carlozamagni/187e478f516cac926682
    :param value:
    :return:
    """
    return struct.unpack('Q', struct.pack('d', value))[0]


# 32位整型的最大值
MAX_INT_32 = 2147483647
# 32位整型的最小值
MIN_INT_32 = -2147483648
DEFAULT_REQUEST_META = num_2_byte_list(0xdabbc200)






class DubboException(RuntimeError):
    pass




class ServiceNotAvailableError(ValueError):
    pass


class HessianTypeError(DubboException):
    """
    Hessian解码的时候发生了解析类型错误
    """
    pass




class DubboResponseException(DubboException):
    """
    dubbo服务端向客户端发送响应时所产生的异常
    """
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


class Object(object):
    """
    创建一个Java对象
    """


    def __init__(self, path, values=None):
        """
        :param path:   Java对象的路径，例如：java.lang.Object
        :param values: 可以在创建对象时就进行赋值
        """
        if not isinstance(path, str):
            raise ValueError('Object path {} should be string type.'.format(path))
        self.__path = path
        if not isinstance(values, dict):
            values = {}
        self.__values = values


    def __getitem__(self, key):
        return self.__values[key]


    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError('Object key {} should be string type.'.format(key))
        self.__values[key] = value


    def __delitem__(self, key):
        del self.__values[key]


    def __repr__(self):
        return '<java object {} at {} with {}>'.format(self.__path, hex(id(self)), self.__values)


    def __contains__(self, key):
        return key in self.__values


    def keys(self):
        return self.__values.keys()


    def get_path(self):
        return self.__path


functions = {}



def ranges(*defined_ranges):
    """
    根据hessian协议，把处理方法交给其定义好的范围
    :param defined_ranges:
    :return:
    """


    def decorator(func):
        # 遍历所有的范围
        for defined_range in defined_ranges:
            if isinstance(defined_range, (tuple, list)):
                if not len(defined_range) == 2:
                    raise ValueError('Invalid range {}'.format(defined_range))
                nums = list(range(defined_range[0], defined_range[1] + 1))
                for num in nums:
                    functions[num] = func
            elif isinstance(defined_range, int):
                functions[defined_range] = func
            else:
                raise ValueError('Defined value {} illegal'.format(defined_ranges))


        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)  # 原始方法正常执行


        return wrapper


    return decorator




class Response(object):
    """
    A class for parsing dubbo response body.
    All types can be parsed:
    * byte
    * boolean
    * int
    * long
    * double
    * string
    * object
    * class
    * generic type
    * list
    * map
    * date
    * null
    """
    def __init__(self, data):
        self.__data = data  # data是字节数组
        self.__index = 0
        self.types = []
        self.objects = []
        # 对于一个类来说，有path的地方就应该有field_name
        self.paths = []
        self.field_names = []


    def get_byte(self):
        """
        获取到头部的字节数据，只是获取并不移动指针
        :return:
        """
        return self.__data[self.__index]


    def length(self):
        """
        当前的字节长度
        :return:
        """
        return len(self.__data) - self.__index


    def read_byte(self):
        """
        读取一个字节并向后移动一位指针
        :return:
        """
        if self.__index >= len(self.__data):
            raise ValueError('Index {} bigger than data length {}'.format(self.__index, len(self.__data)))
        value = self.__data[self.__index]
        self.__index += 1
        return value


    def read_bytes(self, num):
        """
        读取n个字节并向后移动n位指针
        :param num:
        :return:
        """
        value = self.__data[self.__index:self.__index + num]
        self.__index += num
        return value


    @ranges(ord('T'), ord('F'))
    def read_boolean(self):
        """
        读取一个布尔类型
        :return:
        """
        value = self.read_byte()
        if value == ord('T'):
            return True
        elif value == ord('F'):
            return False
        else:
            raise HessianTypeError('illegal boolean value: {0}'.format(value))


    @ranges((0x80, 0xd7), ord('I'))
    def read_int(self):
        """
        读取一个整型数据
        :return:
        """
        value = self.read_byte()
        if 0x80 <= value <= 0xbf:
            result = value - 0x90
        elif 0xc0 <= value <= 0xcf:
            i = (value - 0xc8) << 8
            i |= self.read_byte()
            result = i
        elif 0xd0 <= value <= 0xd7:
            i = (value - 0xd4) << 16
            i |= self.read_byte() << 8
            i |= self.read_byte()
            result = i
        else:
            result = unpack('!i', self.read_bytes(4))[0]
        return result


    @ranges((0x5b, 0x5f), ord('D'))
    def read_double(self):
        """
        读取一个浮点类型
        :return:
        """
        value = self.read_byte()
        if value == 0x5b:
            result = 0.0
        elif value == 0x5c:
            result = 1.0
        elif value == 0x5d:
            result = float(unpack('!b', self.read_bytes(1))[0])
        elif value == 0x5e:
            result = float(unpack('!h', self.read_bytes(2))[0])
        elif value == 0x5f:
            result = float(unpack('!i', self.read_bytes(4))[0]) * 0.001
        elif value == ord('D'):
            result = float(unpack('!d', self.read_bytes(8))[0])
        else:
            raise HessianTypeError('{0} is not a float'.format(value))
        return result


    def _read_utf(self, length):
        """
        读取n个字符
        :param length:
        :return:
        """
        value = ''
        for i in range(length):
            ch = self.read_byte()
            if ch < 0x80:
                value += chr(ch)
            elif (ch & 0xe0) == 0xc0:
                ch1 = self.read_byte()
                value += chr(((ch & 0x1f) << 6) + (ch1 & 0x3f))
            elif (ch & 0xf0) == 0xe0:
                ch1 = self.read_byte()
                ch2 = self.read_byte()
                value += chr(((ch & 0x0f) << 12) + ((ch1 & 0x3f) << 6) + (ch2 & 0x3f))
            else:
                raise ValueError('Can\'t parse utf-8 char {}'.format(ch))
        return value.encode('utf-8')  # 将unicode转化为str类型


    @ranges((0x00, 0x1f), (0x30, 0x33), 0x52, ord('S'))
    def read_string(self):
        """
        读取一个字符串
        :return:
        """
        value = self.read_byte()
        string = ''
        while value == 'R':
            length = unpack('!h', self.read_bytes(2))[0]
            string += self._read_utf(length)
            value = self.read_byte()


        if value == ord('S'):
            length = unpack('!h', self.read_bytes(2))[0]
        elif 0x00 <= value <= 0x1f:
            length = value
        else:
            length = (value - 0x30) << 8 | self.read_byte()


        string += self._read_utf(length).decode('utf-8')
        return string


    @ranges((0x60, 0x6f), ord('O'))
    def read_object(self):
        """
        读取一个对象
        :return:
        """
        result = {}
        self.objects.append(result)
        value = self.read_byte()
        if 0x60 <= value <= 0x6f:
            ref = value - 0x60
        else:
            ref = self.read_int()
        field_names = self.field_names[ref]
        for field_name in field_names:
            field_value = self.read_next()
            result[field_name] = field_value


        path = self.paths[ref]
        if path == 'java.math.BigDecimal':
            result = float(result['value']) or 0
            self.objects[-1] = result
        elif path == 'java.math.BigInteger':
            result = int(result['value'])
            self.objects[-1] = result


        return result


    @ranges(ord('C'))
    def read_class(self):
        """
        读取一个类的类属性，主要是类名和类中的变量名
        :return:
        """
        self.read_byte()
        path = self.read_string()
        self.paths.append(path)


        field_length = self.read_int()
        field_names = []
        for i in range(field_length):
            field_names.append(self.read_string())
        self.field_names.append(field_names)
        return self.read_object()


    def read_type(self):
        """
        type代表了list或者map中泛型的类型，在Python中此类型无意义
        :return:
        """
        _type = self.read_next()
        if isinstance(_type, int):
            return self.types[_type]
        elif isinstance(_type, str):
            self.types.append(_type)
            return _type
        else:
            raise HessianTypeError('Unknown _type type for value: {0}'.format(_type))


    @ranges((0x70, 0x7f), (0x55, 0x58))
    def read_list(self):
        """
        读取一个列表
        :return:
        """
        result = []
        self.objects.append(result)
        value = self.read_byte()
        # 固定长度的有类型短小列表
        if 0x70 <= value <= 0x77:
            _type = self.read_type()  # type对于Python来说没有用处
            length = value - 0x70
            for i in range(length):
                result.append(self.read_next())
        # 固定长度的无类型短小列表
        elif 0x78 <= value <= 0x7f:
            length = value - 0x78
            for i in range(length):
                result.append(self.read_next())
        # 固定长度的有类型列表
        elif value == 0x56:
            _type = self.read_type()
            length = self.read_int()
            for i in range(length):
                result.append(self.read_next())
        # 固定长度的无类型列表
        elif value == 0x58:
            length = self.read_int()
            for i in range(length):
                result.append(self.read_next())
        # 可变长度的有类型列表
        elif value == 0x55:
            _type = self.read_type()
        # 可变长度的无类型列表
        elif value == 0x57:
            pass
        return result


    @ranges((0xd8, 0xff), (0x38, 0x3f), 0x59, ord('L'))
    def read_long(self):
        """
        读取一个long类型的数字
        :return:
        """
        value = self.read_byte()
        if 0xd8 <= value <= 0xef:
            result = value - 0xe0
        elif 0xf0 <= value <= 0xff:
            result = ((value - 0xf8) << 8) | self.read_byte()
        elif 0x38 <= value <= 0x3f:
            i = (value - 0x3c) << 16
            i |= self.read_byte() << 8
            i |= self.read_byte()
            result = i
        elif value == 0x59:
            result = unpack('!i', self.read_bytes(4))[0]
        elif value == ord('L'):
            result = unpack('!q', self.read_bytes(8))[0]
        else:
            raise HessianTypeError('{0} is not long type'.format(value))
        return result


    @ranges(ord('N'))
    def read_null(self):
        """
        读取一个None
        :return:
        """
        value = self.read_byte()
        if value == ord('N'):
            return None
        else:
            raise HessianTypeError('{0} is not null'.format(value))


    @ranges(ord('H'), ord('M'))
    def read_map(self):
        """
        读取一个dict
        :return:
        """
        value = self.read_byte()


        if value == ord('M') or value == ord('H'):
            result = {}
            self.objects.append(result)
            while self.get_byte() != ord('Z'):
                key = self.read_next()
                value = self.read_next()
                result[key] = value
            self.read_byte()  # 干掉最后一个'Z'字符
            return result
        else:
            raise HessianTypeError('{0} is not a map.'.format(value))


    @ranges(0x4a, 0x4b)
    def read_date(self):
        """
        读取一个date类型的值
        :return:
        """
        value = self.read_byte()
        if value == 0x4a:
            timestamp = unpack('!q', self.read_bytes(8))[0]
        elif value == 0x4b:
            timestamp = unpack('!i', self.read_bytes(4))[0]
            timestamp *= 60000
        else:
            raise HessianTypeError('{0} is not date type'.format(value))
        return datetime.fromtimestamp(timestamp / 1e3).strftime("%Y-%m-%dT%H:%M:%S.%f+0800")


    @ranges(0x51)
    def read_ref(self):
        """
        读取一个已知的object/list/map
        :return:
        """
        self.read_byte()  # 干掉0x51
        ref_id = self.read_int()
        return self.objects[ref_id]


    def read_next(self):
        """
        读取下一个变量，自动识别变量类型
        :return:
        """
        data_type = self.get_byte()
        func = functions[data_type]
        return func(self)


    def read_error(self):
        """
        解析Java的错误信息，因为需要知道错误的类型，所以需要单独处理
        :return:
        """
        self.read_byte()
        error_type = self.read_string()
        self.paths.append(error_type)


        field_length = self.read_int()
        field_names = []
        for i in range(field_length):
            field_names.append(self.read_string())
        self.field_names.append(field_names)


        error = self.read_object()
        error['cause'] = error_type
        return error


    def __repr__(self):
        return str(self.__data)






def parse_response_head(response_head):
    """
    对响应头部的字节做解析
    :param response_head:
    :return:
    """
    # Magic number
    if not (response_head[0] == 0xda and response_head[1] == 0xbb):
        raise DubboException('illegal response')


    # 第三位为1表示这是一个心跳包
    if response_head[2] & 0x20 == 0x20:
        if response_head[2] & 0x80 == 0x80:
            # 第一位为1，一个心跳请求的包
            heartbeat = 2
        else:
            # 第一位为0，一个心跳响应的包
            heartbeat = 1
            response_status = response_head[3]
            if response_status != 20:
                raise DubboException(response_status_message[response_status])
    else:
        heartbeat = 0
        response_status = response_head[3]
        if response_status != 20:
            raise DubboResponseException(response_status_message[response_status])
    return heartbeat, unpack('!i', response_head[12:])[0]




class Request(object):
    """
    A class for dumping dubbo request body.
    All types can be dumped:
    * boolean
    * int
    * long
    * double
    * string
    * object
    """
    def __init__(self, request):
        self.__body = request
        self.__classes = []
        self.types = []  # 泛型
        self.invoke_id = 0


    def encode(self):
        """
        把请求序列化为字节数组
        :return:
        """
        request_body = self._encode_request_body()
        invoke_id = list(bytearray(struct.pack('!q', self.invoke_id)))
        request_head = DEFAULT_REQUEST_META + invoke_id + get_request_body_length(request_body)
        return bytearray(request_head + request_body)


    def _get_parameter_types(self, arguments):
        """
        针对所有的参数计算得到参数类型字符串
        :param arguments:
        :return:
        """
        parameter_types = ''
        # 判断并得出参数的类型
        for argument in arguments:
            parameter_types += self._get_class_name(argument)
        return parameter_types


    def _get_class_name(self, _class):
        """
        根据一个字段的类型得到其在Java中对应类的全限定名
        转换规则：https://stackoverflow.com/a/3442100/4614538
        :param _class:
        :return:
        """
        if isinstance(_class, bool):  # bool类型的判断必须放在int类型判断的前面
            return 'Z'
        elif isinstance(_class, int):
            if MIN_INT_32 <= _class <= MAX_INT_32:
                return 'I'
            else:
                return 'J'
        elif isinstance(_class, float):
            return 'D'
        elif isinstance(_class, str):
            return 'L' + 'java/lang/String' + ';'
        elif isinstance(_class, Object):
            path = _class.get_path()
            path = 'L' + path.replace('.', '/') + ';'
            return path
        elif isinstance(_class, list):
            if len(_class) == 0:
                raise HessianTypeError('Method parameter {} is a list but length is zero'.format(_class))
            return '[' + self._get_class_name(_class[0])
        else:
            raise HessianTypeError('Unknown argument type: {0}'.format(_class))


    def _encode_request_body(self):
        """
        对所有的已知的参数根据dubbo协议进行编码
        :return:
        """
        dubbo_version = self.__body['dubbo_version']
        path = self.__body['path']
        version = self.__body['version']
        method = self.__body['method']
        arguments = self.__body['arguments']


        body = []
        body.extend(self._encode_single_value(dubbo_version))
        body.extend(self._encode_single_value(path))
        body.extend(self._encode_single_value(version))
        body.extend(self._encode_single_value(method))
        body.extend(self._encode_single_value(arguments))
        for argument in arguments:
            body.extend(self._encode_single_value(argument))

        attachments = {
            'path': path,
            'interface': path,
            'version': version
        }
        # attachments参数以H开头，以Z结尾
        body.append(ord('H'))
        for key in attachments.keys():
            value = attachments[key]
            body.extend(self._encode_single_value(key))
            body.extend(self._encode_single_value(value))
        body.append(ord('Z'))


        # 因为在上面的逻辑中没有对byte大小进行检测，所以在这里进行统一的处理
        for i in range(len(body)):
            body[i] = body[i] & 0xff
        return body


    @staticmethod
    def _encode_bool(value):
        """
        对bool类型进行编码
        :param value:
        :return:
        """
        result = []
        if value:
            result.append(ord('T'))
        else:
            result.append(ord('F'))
        return result


    @staticmethod
    def _encode_int(value):
        """
        对整数进行编码
        :param value:
        :return:
        """
        result = []
        # 超出int类型范围的值则转化为long类型
        # 这里问题在于对于落在int范围内的数字，我们无法判断其是long类型还是int类型，所以一律认为其是int类型
        if value > MAX_INT_32 or value < MIN_INT_32:
            result.append(ord('L'))
            result.extend(list(bytearray(struct.pack('!q', value))))
            return result


        if -0x10 <= value <= 0x2f:
            result.append(value + 0x90)
        elif -0x800 <= value <= 0x7ff:
            result.append(0xc8 + (value >> 8))
            result.append(value)
        elif -0x40000 <= value <= 0x3ffff:
            result.append(0xd4 + (value >> 16))
            result.append(value >> 8)
            result.append(value)
        else:
            result.append(ord('I'))
            result.append(value >> 24)
            result.append(value >> 16)
            result.append(value >> 8)
            result.append(value)
        return result


    @staticmethod
    def _encode_float(value):
        """
        对浮点类型进行编码
        :param value:
        :return:
        """
        result = []
        int_value = int(value)
        if int_value == value:
            if int_value == 0:
                result.append(0x5b)
                return result
            elif int_value == 1:
                result.append(0x5c)
                return result
            elif -0x80 <= int_value < 0x80:
                result.append(0x5d)
                result.append(int_value)
                return result
            elif -0x8000 <= int_value < 0x8000:
                result.append(0x5e)
                result.append(int_value >> 8)
                result.append(int_value)
                return result


        mills = int(value * 1000)
        if 0.001 * mills == value and MIN_INT_32 <= mills <= MAX_INT_32:
            result.append(0x5f)
            result.append(mills >> 24)
            result.append(mills >> 16)
            result.append(mills >> 8)
            result.append(mills)
            return result


        bits = double_to_long_bits(value)
        result.append(ord('D'))
        result.append(bits >> 56)
        result.append(bits >> 48)
        result.append(bits >> 40)
        result.append(bits >> 32)
        result.append(bits >> 24)
        result.append(bits >> 16)
        result.append(bits >> 8)
        result.append(bits)
        return result


    @staticmethod
    def _encode_utf(value):
        """
        对字符串进行编码，编码格式utf-8
        参见方法：com.alibaba.com.caucho.hessian.io.Hessian2Output#printString
        :param value:
        :return:
        """
        result = []
        for v in value:
            ch = ord(v)
            if ch < 0x80:
                result.append(ch & 0xff)
            elif ch < 0x800:
                result.append((0xc0 + ((ch >> 6) & 0x1f)) & 0xff)
                result.append((0x80 + (ch & 0x3f)) & 0xff)
            else:
                result.append((0xe0 + ((ch >> 12) & 0xf)) & 0xff)
                result.append((0x80 + ((ch >> 6) & 0x3f)) & 0xff)
                result.append((0x80 + (ch & 0x3f)) & 0xff)
        return result


    def _encode_str(self, value):
        """
        对一个字符串进行编码
        :param value:
        :return:
        """
        result = []
        # 在进行网络传输操作时一律使用unicode进行操作
        if isinstance(value, str):
            value = value.encode().decode('utf-8')
        length = len(value)
        if length <= 0x1f:
            result.append(0x00 + length)
        elif length <= 0x3ff:
            result.append(0x30 + (length >> 8))
            result.append(length)
        else:
            result.append(ord('S'))
            result.append(length >> 8)
            result.append(length)


        result.extend(self._encode_utf(value))
        return result


    def _encode_object(self, value):
        """
        对一个对象进行编码
        :param value:
        :return:
        """
        result = []
        path = value.get_path()
        field_names = value.keys()


        if path not in self.__classes:
            result.append(ord('C'))
            result.extend(self._encode_single_value(path))


            result.extend(self._encode_single_value(len(field_names)))


            for field_name in field_names:
                print(f"field_name: {field_name}")
                result.extend(self._encode_single_value(field_name))
            self.__classes.append(path)
        class_id = self.__classes.index(path)
        if class_id <= 0xf:
            class_id += 0x60
            class_id &= 0xff
            result.append(class_id)
        else:
            result.append(ord('O'))
            result.extend(self._encode_single_value(class_id))
        for field_name in field_names:
            result.extend(self._encode_single_value(value[field_name]))
        return result


    def _encode_list(self, value):
        """
        对一个列表进行编码
        :param value:
        :return:
        """
        result = []
        length = len(value)
        if length == 0:
            # 没有值则无法判断类型，一律返回null
            return self._encode_single_value(None)
        if isinstance(value[0], bool):
            _type = '[boolean'
        elif isinstance(value[0], int):
            _type = '[int'
        elif isinstance(value[0], float):
            _type = '[double'
        elif isinstance(value[0], str):
            _type = '[string'
        elif isinstance(value[0], Object):
            _type = '[object'
        else:
            raise HessianTypeError('Unknown list type: {}'.format(value[0]))
        if length < 0x7:
            result.append(0x70 + length)
            if _type not in self.types:
                self.types.append(_type)
                result.extend(self._encode_single_value(_type))
            else:
                result.extend(self._encode_single_value(self.types.index(_type)))
        else:
            result.append(0x56)
            if _type not in self.types:
                self.types.append(_type)
                result.extend(self._encode_single_value(_type))
            else:
                result.extend(self._encode_single_value(self.types.index(_type)))
            result.extend(self._encode_single_value(length))
        for v in value:
            if type(value[0]) != type(v):
                raise HessianTypeError('All elements in list must be the same type, first type'
                                       ' is {0} but current type is {1}'.format(type(value[0]), type(v)))
            result.extend(self._encode_single_value(v))
        return result


    def _encode_single_value(self, value):
        """
        根据hessian协议对单个变量进行编码
        :param value:
        :return:
        """
        # 布尔类型
        if isinstance(value, bool):
            return self._encode_bool(value)
        # 整型（包括长整型）
        elif isinstance(value, int):
            return self._encode_int(value)
        # 浮点类型
        elif isinstance(value, float):
            return self._encode_float(value)
        # 字符串类型
        elif isinstance(value, str):
            return self._encode_str(value)
        # 对象类型
        elif isinstance(value, Object):
            return self._encode_object(value)
        # 列表(list)类型，不可以使用tuple替代
        elif isinstance(value, list):
            return self._encode_list(value)
        # null
        elif value is None:
            return [ord('N')]
        else:
            raise HessianTypeError('Unknown argument type: {}'.format(value))






class socketDubbo:
    def __init__(self, interface, host_port, zkClient=None):
        self.host, self.port = host_port.split(":")
        self.interface = interface
        self.zk = zkClient
        self.__initial_connect()
    
    def __initial_connect(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.conn.settimeout(120)
        self.conn.setblocking(True)
    
    def __call__(self, method, method_args):
        return self.request(method, method_args)

    def request(self, method, method_args):
        # 获取代理地址
        if self.zk is not None:
            self._get_zk_host_port()

        # 建立socket连接
        self.conn.connect((self.host, int(self.port)))
        # hession 序列化编码
        cmd_ = self.generate_cmd(method_args)
        self.conn.send(cmd_)
        res = self.conn.recv(1024)
        res_b = list(bytearray(res))
        data = bytearray(res_b)
        rp = Response(data[16:])
        flag = rp.read_int()
        res_f = rp.read_next()
        return res_f

    def generate_cmd(self, method_args):
        request_param = {
            'dubbo_version': '2.0.2',
            'version': 'dev',
            'path': self.interface,
            'method': '$invoke',
            'arguments': [json.dumps(method_args)]
        }
        rq_ = Request(request_param) 
        request_data = rq_.encode() 
        return request_data

    def _get_zk_host_port(self):
        address_list = self.zk.get_service_address(self.interface)
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
    a = datetime.strptime('2022-05-16 19:50:00', "%Y-%m-%d %H:%M:%S")
    serv_ = 'xx.xxxx.xx.xx.api.service.Xxxxx'
    zk = Zookeeper(host_dict['dev'])
    db_ = Dubbo(
        interface=serv_,
        host_port=host_dict['dev'], 
        zkClient=zk
    )
    res_ = db_.request(method='xxx', method_args=method_args)
    print("res_: \n")






