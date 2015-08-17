import unittest

import env
from discretizer import BaseDiscretizer, DiscretizerException, \
    LinearDiscretizer, CubeRootDiscretizer, SigmoidDiscretizer


# to run tests from the repository root directory:
# > python -m unittest discover -s tests -v

# with coverage.py:
# > coverage run -m unittest discover -s tests -v; coverage report -m


class TestConversions(unittest.TestCase):
    def test_bytearray_to_bits(self):
        # 1 byte
        ba = bytearray([255])
        bits = BaseDiscretizer.bytearray_to_bits(ba)
        self.assertEqual(bits, '0b11111111')

        # 4 bytes
        ba = bytearray([0, 255, 127, 3])
        bits = BaseDiscretizer.bytearray_to_bits(ba)
        self.assertEqual(bits, '0b'+'00000000'+'11111111'+'01111111'+'00000011')

        # 0 bytes
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bytearray_to_bits,
                          bytearray())

        # not a bytearray
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bytearray_to_bits,
                          0.0)
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bytearray_to_bits,
                          [0, 1, 2])

    def tests_bits_to_bytearray(self):
        # 1 byte
        ba = BaseDiscretizer.bits_to_bytearray('0b11111111')
        self.assertIsInstance(ba, bytearray)
        self.assertEqual(len(ba), 1)
        self.assertEqual(ba[0], 255)

        # 1 byte, shortened
        ba = BaseDiscretizer.bits_to_bytearray('0b10')
        self.assertEqual(len(ba), 1)
        self.assertEqual(ba[0], 2)

        # 4 bytes
        ba = BaseDiscretizer.bits_to_bytearray('0b'+'11111111'+'00000000'+'01111111'+'00000011')
        self.assertEqual(len(ba), 4)
        self.assertEqual(ba, bytearray([255, 0, 127, 3]))

        # 4 bytes (zero first)
        ba = BaseDiscretizer.bits_to_bytearray('0b'+'00000000'+'11111111'+'01111111'+'00000011')
        self.assertEqual(len(ba), 4)
        self.assertEqual(ba, bytearray([0, 255, 127, 3]))

        # 4 bytes, shortened
        ba = BaseDiscretizer.bits_to_bytearray('0b'+'00'+'11111111'+'01111111'+'00000011')
        self.assertEqual(len(ba), 4)
        self.assertEqual(ba, bytearray([0, 255, 127, 3]))

        # not a string
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bytearray,
                          1101)

        # empty string
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bytearray,
                          '')

        # not a string of bits (should begin with '0b')
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bytearray,
                          '11111111')

        # no bits
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bytearray,
                          '0b')

        # invalid bits
        self.assertRaises(ValueError,
                          BaseDiscretizer.bits_to_bytearray,
                          '0b123')

    def test_bucket_num_to_bits(self):
        # valid
        bits = BaseDiscretizer.bucket_num_to_bits(0)
        self.assertEqual(bits, '0b00000000')
        bits = BaseDiscretizer.bucket_num_to_bits(3)
        self.assertEqual(bits, '0b00000011')
        bits = BaseDiscretizer.bucket_num_to_bits(255)
        self.assertEqual(bits, '0b11111111')
        bits = BaseDiscretizer.bucket_num_to_bits(512)
        self.assertEqual(bits, '0b'+'00000010'+'00000000')
        bits = BaseDiscretizer.bucket_num_to_bits(16744195)
        self.assertEqual(bits, '0b'+'11111111'+'01111111'+'00000011')

        # not an integer
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bucket_num_to_bits,
                          'abcd')

        # bad bucket number
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bucket_num_to_bits,
                          -1)

    def test_bits_to_bucket_num(self):
        # valid
        bn = BaseDiscretizer.bits_to_bucket_num('0b0')
        self.assertEqual(bn, 0)
        bn = BaseDiscretizer.bits_to_bucket_num('0b10')
        self.assertEqual(bn, 2)
        bn = BaseDiscretizer.bits_to_bucket_num('0b0010')
        self.assertEqual(bn, 2)
        bn = BaseDiscretizer.bits_to_bucket_num('0b00000000010')
        self.assertEqual(bn, 2)

        # not a string
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bucket_num,
                          1101)

        # empty string
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bucket_num,
                          '')

        # not a string of bits (should begin with '0b')
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bucket_num,
                          '11111111')

        # no bits
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bits_to_bucket_num,
                          '0b')

        # invalid bits
        self.assertRaises(ValueError,
                          BaseDiscretizer.bits_to_bucket_num,
                          '0b123')

    def test_bucket_num_to_bytearray(self):
        # valid
        ba = BaseDiscretizer.bucket_num_to_bytearray(0)
        self.assertIsInstance(ba, bytearray)
        self.assertEqual(len(ba), 1)
        self.assertEqual(ba[0], 0)

        ba = BaseDiscretizer.bucket_num_to_bytearray(3)
        self.assertEqual(ba, bytearray([3]))

        ba = BaseDiscretizer.bucket_num_to_bytearray(255)
        self.assertEqual(ba, bytearray([255]))

        ba = BaseDiscretizer.bucket_num_to_bytearray(256)
        self.assertEqual(ba, bytearray([1, 0]))

        ba = BaseDiscretizer.bucket_num_to_bytearray(257)
        self.assertEqual(ba, bytearray([1, 1]))

        ba = BaseDiscretizer.bucket_num_to_bytearray(65535)
        self.assertEqual(ba, bytearray([255, 255]))

        ba = BaseDiscretizer.bucket_num_to_bytearray(65536)
        self.assertEqual(ba, bytearray([1, 0, 0]))

        # not an integer
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bucket_num_to_bits,
                          'abcd')

        # bad bucket number
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bucket_num_to_bits,
                          -1)

    def test_bytearray_to_bucket_num(self):
        # valid
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([0]))
        self.assertIsInstance(bn, int)
        self.assertEqual(bn, 0)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([0, 0, 0]))
        self.assertEqual(bn, 0)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([3]))
        self.assertEqual(bn, 3)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([255]))
        self.assertEqual(bn, 255)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([1, 0]))
        self.assertEqual(bn, 256)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([1, 1]))
        self.assertEqual(bn, 257)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([255, 255]))
        self.assertEqual(bn, 65535)
        bn = BaseDiscretizer.bytearray_to_bucket_num(bytearray([1, 0, 0]))
        self.assertEqual(bn, 65536)

        # 0 bytes
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bytearray_to_bucket_num,
                          bytearray())

        # not a bytearray
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bytearray_to_bucket_num,
                          0.0)
        self.assertRaises(DiscretizerException,
                          BaseDiscretizer.bytearray_to_bucket_num,
                          [0, 1, 2])


class TestLinearDiscretizer(unittest.TestCase):
    def test_basics(self):
        # valid, 1 byte
        d = LinearDiscretizer(1, -10.0, 20.0)
        self.assertEqual(d.num_bytes, 1)
        self.assertEqual(d.num_buckets, 256)
        self.assertEqual(d.max_bucket, 255)
        self.assertAlmostEqual(d.max_bucket_float, 255.0)
        self.assertAlmostEqual(d.val_min, -10.0)
        self.assertAlmostEqual(d.val_max, 20.0)
        self.assertAlmostEqual(d.val_range, 30.0)

        # valid, 3 bytes
        d = LinearDiscretizer(3, -10.0, 20.0)
        self.assertEqual(d.num_bytes, 3)
        self.assertEqual(d.num_buckets, 16777216)
        self.assertEqual(d.max_bucket, 16777215)
        self.assertAlmostEqual(d.max_bucket_float, 16777215.0)
        self.assertAlmostEqual(d.val_min, -10.0)
        self.assertAlmostEqual(d.val_max, 20.0)
        self.assertAlmostEqual(d.val_range, 30.0)

        # invalid cases
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          '', -10.0, 20.0)
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          0, -10.0, 20.0)
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          -5, -10.0, 20.0)
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          8, -10.0, 20.0)
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          1, '', 20.0)
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          1, -10.0, '')
        self.assertRaises(DiscretizerException, LinearDiscretizer,
                          1, 20.0, -10.0)

    def test_mapper(self):
        d = LinearDiscretizer(1, -10.0, 20.0)

        # base encoder
        self.assertAlmostEqual(d.map_encoder(0.00), 0.00)
        self.assertAlmostEqual(d.map_encoder(0.25), 0.25)
        self.assertAlmostEqual(d.map_encoder(0.50), 0.50)
        self.assertAlmostEqual(d.map_encoder(0.75), 0.75)
        self.assertAlmostEqual(d.map_encoder(1.00), 1.00)

        # base decoder
        self.assertAlmostEqual(d.map_decoder(0.00), 0.00)
        self.assertAlmostEqual(d.map_decoder(0.25), 0.25)
        self.assertAlmostEqual(d.map_decoder(0.50), 0.50)
        self.assertAlmostEqual(d.map_decoder(0.75), 0.75)
        self.assertAlmostEqual(d.map_decoder(1.00), 1.00)

    def test_bucket_num(self):
        d = LinearDiscretizer(1, -10.0, 20.0)

        # to bucket number, valid cases
        self.assertEqual(d.val_to_bucket_num(-11.0),   0)
        self.assertEqual(d.val_to_bucket_num(-10.0),   0)
        self.assertEqual(d.val_to_bucket_num( -2.5),  64)
        self.assertEqual(d.val_to_bucket_num(  4.9), 127)
        self.assertEqual(d.val_to_bucket_num(  5.0), 128)
        self.assertEqual(d.val_to_bucket_num( 12.5), 191)
        self.assertEqual(d.val_to_bucket_num( 20.0), 255)
        self.assertEqual(d.val_to_bucket_num( 21.0), 255)

        # to bucket number, invalid cases
        self.assertRaises(DiscretizerException, d.val_to_bucket_num, '')
        self.assertRaises(DiscretizerException, d.val_to_bucket_num, 1)

        # from bucket number, valid cases
        self.assertAlmostEqual(d.bucket_num_to_val(  0), -10.0000, 4)
        self.assertAlmostEqual(d.bucket_num_to_val( 64),  -2.4706, 4)
        self.assertAlmostEqual(d.bucket_num_to_val(127),   4.9412, 4)
        self.assertAlmostEqual(d.bucket_num_to_val(128),   5.0588, 4)
        self.assertAlmostEqual(d.bucket_num_to_val(191),  12.4706, 4)
        self.assertAlmostEqual(d.bucket_num_to_val(255),  20.0000, 4)

        # from bucket number, invalid cases
        self.assertRaises(DiscretizerException, d.bucket_num_to_val, '')
        self.assertRaises(DiscretizerException, d.bucket_num_to_val, 1.1)
        self.assertRaises(DiscretizerException, d.bucket_num_to_val, -1)
        self.assertRaises(DiscretizerException, d.bucket_num_to_val, 256)

    def test_encdec_small(self):
        d = LinearDiscretizer(1, -10.0, 20.0)

        # encoder, valid cases
        self.assertEqual(d.encode(-11.0), bytearray([  0]))
        self.assertEqual(d.encode(-10.0), bytearray([  0]))
        self.assertEqual(d.encode( -2.5), bytearray([ 64]))
        self.assertEqual(d.encode(  4.9), bytearray([127]))
        self.assertEqual(d.encode(  5.0), bytearray([128]))
        self.assertEqual(d.encode( 12.5), bytearray([191]))
        self.assertEqual(d.encode( 20.0), bytearray([255]))
        self.assertEqual(d.encode( 21.0), bytearray([255]))

        # encoder, invalid cases
        self.assertRaises(DiscretizerException, d.encode, '')
        self.assertRaises(DiscretizerException, d.encode, 1)

        # decoder, valid cases
        self.assertAlmostEqual(d.decode(bytearray([  0])), -10.0000, 4)
        self.assertAlmostEqual(d.decode(bytearray([ 64])),  -2.4706, 4)
        self.assertAlmostEqual(d.decode(bytearray([127])),   4.9412, 4)
        self.assertAlmostEqual(d.decode(bytearray([128])),   5.0588, 4)
        self.assertAlmostEqual(d.decode(bytearray([191])),  12.4706, 4)
        self.assertAlmostEqual(d.decode(bytearray([255])),  20.0000, 4)

        # decoder, invalid cases
        self.assertRaises(DiscretizerException, d.decode, '')
        self.assertRaises(DiscretizerException, d.decode, 1.1)
        self.assertRaises(DiscretizerException, d.decode, bytearray())
        self.assertRaises(DiscretizerException, d.decode, bytearray([0, 0]))

    def test_encdec_big(self):
        d = LinearDiscretizer(3, 0.0, 100.0)
        self.assertEqual(d.num_bytes, 3)
        self.assertEqual(d.num_buckets, 16777216)
        self.assertEqual(d.max_bucket, 16777215)

        # encoder, valid cases
        self.assertEqual(d.encode( -1.0), bytearray([  0,   0,   0]))
        self.assertEqual(d.encode(  0.0), bytearray([  0,   0,   0]))
        self.assertEqual(d.encode( 25.0), bytearray([ 64,   0,   0]))
        self.assertEqual(d.encode( 50.0), bytearray([128,   0,   0]))
        self.assertEqual(d.encode( 75.0), bytearray([191, 255, 255]))
        self.assertEqual(d.encode(100.0), bytearray([255, 255, 255]))
        self.assertEqual(d.encode(101.0), bytearray([255, 255, 255]))

        # encoder, byte transitions
        self.assertEqual(d.encode(0.00152), bytearray([  0,   0, 255]))
        self.assertEqual(d.encode(0.39062), bytearray([  0, 255, 255]))

        # decoder, valid cases
        self.assertAlmostEqual(d.decode(bytearray([  0,   0,   0])),   0.00000, 5)
        self.assertAlmostEqual(d.decode(bytearray([  0,   0, 255])),   0.00152, 5)
        self.assertAlmostEqual(d.decode(bytearray([  0, 255, 255])),   0.39062, 5)
        self.assertAlmostEqual(d.decode(bytearray([ 64,   0,   0])),  25.00000, 5)
        self.assertAlmostEqual(d.decode(bytearray([128,   0,   0])),  50.00000, 5)
        self.assertAlmostEqual(d.decode(bytearray([191, 255, 255])),  75.00000, 5)
        self.assertAlmostEqual(d.decode(bytearray([255, 255, 255])), 100.00000, 5)


class TestCubeRootDiscretizer(unittest.TestCase):
    def test_mapper(self):
        d = CubeRootDiscretizer(1, -10.0, 20.0)

        # encoder
        self.assertAlmostEqual(d.map_encoder(0.00), 0.00000, 5)
        self.assertAlmostEqual(d.map_encoder(0.25), 0.10315, 5)
        self.assertAlmostEqual(d.map_encoder(0.50), 0.50000, 5)
        self.assertAlmostEqual(d.map_encoder(0.75), 0.89685, 5)
        self.assertAlmostEqual(d.map_encoder(1.00), 1.00000, 5)

        # decoder
        self.assertAlmostEqual(d.map_decoder(0.00000), 0.00000, 5)
        self.assertAlmostEqual(d.map_decoder(0.10315), 0.25000, 5)
        self.assertAlmostEqual(d.map_decoder(0.50000), 0.50000, 5)
        self.assertAlmostEqual(d.map_decoder(0.89685), 0.75000, 5)
        self.assertAlmostEqual(d.map_decoder(1.00000), 1.00000, 5)


class TestSigmoidDiscretizer(unittest.TestCase):
    def test_mapper(self):
        d = SigmoidDiscretizer(1, -10.0, 20.0, 20.0)

        # encoder
        self.assertAlmostEqual(d.map_encoder(0.00), 0.0000000, 6)
        self.assertAlmostEqual(d.map_encoder(0.25), 0.0066481, 6)
        self.assertAlmostEqual(d.map_encoder(0.40), 0.1191683, 6)
        self.assertAlmostEqual(d.map_encoder(0.50), 0.5000000, 6)
        self.assertAlmostEqual(d.map_encoder(0.60), 0.8808317, 6)
        self.assertAlmostEqual(d.map_encoder(0.75), 0.9933520, 6)
        self.assertAlmostEqual(d.map_encoder(1.00), 1.0000000, 6)

        # decoder
        self.assertAlmostEqual(d.map_decoder(0.0000000), 0.00000, 6)
        self.assertAlmostEqual(d.map_decoder(0.0066481), 0.25000, 5)
        self.assertAlmostEqual(d.map_decoder(0.1191683), 0.40000, 5)
        self.assertAlmostEqual(d.map_decoder(0.5000000), 0.50000, 5)
        self.assertAlmostEqual(d.map_decoder(0.8808317), 0.60000, 5)
        self.assertAlmostEqual(d.map_decoder(0.9933520), 0.75000, 5)
        self.assertAlmostEqual(d.map_decoder(1.0000000), 1.00000, 6)

    def test_params(self):
        self.assertRaises(DiscretizerException, SigmoidDiscretizer,
                          1, 0.0, 1.0, '')
        self.assertRaises(DiscretizerException, SigmoidDiscretizer,
                          1, 0.0, 1.0, 0.0)
        self.assertRaises(DiscretizerException, SigmoidDiscretizer,
                          1, 0.0, 1.0, -1.0)


if __name__ == '__main__':
    unittest.main()
