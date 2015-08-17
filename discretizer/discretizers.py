import math


ONE_THIRD = 1.0/3.0


class DiscretizerException(Exception):
    pass


class BaseDiscretizer(object):
    def __init__(self, num_bytes, val_min, val_max):
        if not isinstance(num_bytes, int):
            raise DiscretizerException('Number of bytes must be an integer.')
        if num_bytes <= 0:
            raise DiscretizerException('Number of bytes must be > 0.')
        if num_bytes >= 8:
            raise DiscretizerException('Too many bytes, use a 64-bit double '
                                       'instead.')
        self._num_bytes = num_bytes
        num_bits = 8 * self._num_bytes
        self._num_buckets = 2 ** num_bits
        self._max_bucket = self._num_buckets - 1
        self._max_bucket_float = float(self._max_bucket)
        if not isinstance(val_min, float):
            raise DiscretizerException('Minimum value must be a float.')
        if not isinstance(val_max, float):
            raise DiscretizerException('Maximum value must be a float.')
        if val_max <= val_min:
            raise DiscretizerException('Max/min values invalid.')
        self._val_min = val_min
        self._val_max = val_max
        self._val_range = self._val_max - self._val_min
        assert self._val_range > 0.0

    @property
    def num_bytes(self):
        return self._num_bytes

    @property
    def num_buckets(self):
        return self._num_buckets

    @property
    def max_bucket(self):
        return self._max_bucket

    @property
    def max_bucket_float(self):
        return self._max_bucket_float

    @property
    def val_min(self):
        return self._val_min

    @property
    def val_max(self):
        return self._val_max

    @property
    def val_range(self):
        return self._val_range

    def encode(self, val):
        bucket_num = self.val_to_bucket_num(val)
        ba = self.bucket_num_to_bytearray(bucket_num)
        len_ba = len(ba)
        if len_ba != self.num_bytes:
            ba.reverse()
            for i in range(self.num_bytes - len_ba):
                ba.append(0)
            ba.reverse()
        return ba

    def decode(self, ba):
        if not isinstance(ba, bytearray):
            raise DiscretizerException('Input not bytearray.')
        if len(ba) != self.num_bytes:
            raise DiscretizerException('Invalid number of bytes parsed.')
        bucket_num = self.bytearray_to_bucket_num(ba)
        return self.bucket_num_to_val(bucket_num)

    def val_to_bucket_num(self, val):
        if not isinstance(val, float):
            raise DiscretizerException('Value must be a float.')

        # normalise the input value and check bounds (v = [0.0, 1.0])
        v = (float(val) - self.val_min) / self.val_range
        if v <= 0.0:
            return 0
        elif v >= 1.0:
            return self.max_bucket

        # execute mapping function
        b = float(self.map_encoder(v))

        # get nearest bucket number, clamp and return
        bucket_num = int(round(b * self.max_bucket))
        if bucket_num < 0:
            bucket_num = 0
        elif bucket_num > self.max_bucket:
            bucket_num = self.max_bucket
        return bucket_num

    def bucket_num_to_val(self, bucket_num):
        if not isinstance(bucket_num, int):
            raise DiscretizerException('Bucket number must be an integer.')
        if bucket_num < 0:
            raise DiscretizerException('Bucket number must be >= 0.')
        if bucket_num > self.max_bucket:
            raise DiscretizerException('Bucket number must be <= maximum.')

        # compute bucket factor and check bounds (b = [0.0, 1.0])
        b = float(bucket_num) / self.max_bucket_float
        if b <= 0.0:
            return self.val_min
        elif b >= 1.0:
            return self.val_max

        # execute mapping function
        v = float(self.map_decoder(b))

        # clamp, compute value and return value
        if v <= 0.0:
            return self.val_min
        elif v >= 1.0:
            return self.val_max
        val = self.val_min + v * self.val_range
        return val

    @staticmethod
    def bucket_num_to_bytearray(bucket_num):
        bits = BaseDiscretizer.bucket_num_to_bits(bucket_num)
        return BaseDiscretizer.bits_to_bytearray(bits)

    @staticmethod
    def bytearray_to_bucket_num(ba):
        bits = BaseDiscretizer.bytearray_to_bits(ba)
        return BaseDiscretizer.bits_to_bucket_num(bits)

    @staticmethod
    def bucket_num_to_bits(bucket_num):
        if not isinstance(bucket_num, int):
            raise DiscretizerException('Bucket number must be an integer.')
        if bucket_num < 0:
            raise DiscretizerException('Bucket number must be >= 0.')
        bits = bin(bucket_num).replace('0b', '')
        len_bits = len(bits)
        assert len_bits > 0
        num_bytes = int(len_bits / 8)
        if (len_bits % 8) > 0:
            num_bytes += 1
        bits = '0b' + bits.zfill(num_bytes * 8)
        return bits

    @staticmethod
    def bits_to_bucket_num(bits):
        if not isinstance(bits, str):
            raise DiscretizerException('Input not a string.')
        if bits[0:2] != '0b':
            raise DiscretizerException('Input not a string of bits.')
        bits = bits.replace('0b', '')
        if len(bits) <= 0:
            raise DiscretizerException('No input bits.')
        bucket_num = int(bits, 2)
        assert bucket_num >= 0
        return bucket_num

    @staticmethod
    def bytearray_to_bits(ba):
        if not isinstance(ba, bytearray):
            raise DiscretizerException('Input not a bytearray.')
        if len(ba) <= 0:
            raise DiscretizerException('Bytearray is empty.')
        bits = '0b'
        for i in ba:
            assert i >= 0 and i <= 255
            bits += (bin(i).replace('0b', '').zfill(8))
        return bits

    @staticmethod
    def bits_to_bytearray(bits):
        if not isinstance(bits, str):
            raise DiscretizerException('Input not a string.')
        if bits[0:2] != '0b':
            raise DiscretizerException('Input not a string of bits.')
        bits = bits.replace('0b', '')
        len_bits = len(bits)
        if len_bits <= 0:
            raise DiscretizerException('No input bits.')
        num_bytes = int(len_bits / 8)
        if (len_bits % 8) > 0:
            num_bytes += 1
            bits = bits.zfill(num_bytes * 8)
        ba = bytearray()
        for i in range(num_bytes):
            local_bits = bits[i*8:(i+1)*8]
            ba.append(int(local_bits, 2))
        return ba


class LinearDiscretizer(BaseDiscretizer):
    def __init__(self, num_bytes, val_min, val_max):
        BaseDiscretizer.__init__(self, num_bytes, val_min, val_max)

    def map_encoder(self, v):
        return v

    def map_decoder(self, b):
        return b


class CubeRootDiscretizer(BaseDiscretizer):
    def __init__(self, num_bytes, val_min, val_max):
        BaseDiscretizer.__init__(self, num_bytes, val_min, val_max)

    def map_encoder(self, v):
        # compute only the real cube root
        x = (v - 0.5) * 0.25
        b = (math.pow(abs(x), ONE_THIRD) * (1, -1)[x < 0.0]) + 0.5
        return b

    def map_decoder(self, b):
        v = 4.0 * math.pow(b - 0.5, 3.0) + 0.5
        return v


class SigmoidDiscretizer(BaseDiscretizer):
    def __init__(self, num_bytes, val_min, val_max, sharpness):
        BaseDiscretizer.__init__(self, num_bytes, val_min, val_max)
        if not isinstance(sharpness, float):
            raise DiscretizerException('Sharpness must be a float.')
        if sharpness <= 0.0:
            raise DiscretizerException('Sharpness must be > 0.')
        self._k = sharpness
        self._inv_k = 1.0 / self._k
        self._S = 2.0 / (math.exp(0.5*self._k) - 1.0)
        self._one_plus_S = 1.0 + self._S
        self._half_S = 0.5 * self._S

    def map_encoder(self, v):
        f = 1.0 + math.exp(self._k * (0.5 - v))
        b = self._one_plus_S / f - self._half_S
        return b

    def map_decoder(self, b):
        f = self._one_plus_S / (b + self._half_S) - 1.0
        v = 0.5 - self._inv_k * math.log(f)
        return v
