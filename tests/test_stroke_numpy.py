

from stroke import Stroke
from unittest import TestCase
import numpy as np


class TestStrokeNumpyMath(TestCase):

    def setUp(self):

        self.x = np.linspace(-1, 1, 10)
        self.y = np.exp(self.x) + np.cos(np.pi * self.x) - 1

        self.f1 = Stroke(self.x, self.y, "linear", "poly")
        self.f2 = Stroke(self.x, self.y, "quadratic", "poly")
        self.f3 = Stroke(self.x, self.y, "cubic", "poly")

        self.xother = np.linspace(-1, 1, 10)
        self.yother = np.sin(2 * self.xother)

        self.fother = Stroke(self.xother, self.yother, "linear", "poly")
    
    def test_add(self):
        
        y1 = np.add(self.f1, 5)(self.x)
        y2 = np.add(self.f2, 5)(self.x)
        y3 = np.add(self.f3, 5)(self.x)

        val = np.add(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.add(self.f1, self.fother)(self.x)
        yother2 = np.add(self.f2, self.fother)(self.x)
        yother3 = np.add(self.f3, self.fother)(self.x)

        val = np.add(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_subtract(self):
        
        y1 = np.subtract(self.f1, 5)(self.x)
        y2 = np.subtract(self.f2, 5)(self.x)
        y3 = np.subtract(self.f3, 5)(self.x)

        val = np.subtract(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.subtract(self.f1, self.fother)(self.x)
        yother2 = np.subtract(self.f2, self.fother)(self.x)
        yother3 = np.subtract(self.f3, self.fother)(self.x)

        val = np.subtract(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_multiply(self):
        
        y1 = np.multiply(self.f1, 5)(self.x)
        y2 = np.multiply(self.f2, 5)(self.x)
        y3 = np.multiply(self.f3, 5)(self.x)

        val = np.multiply(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.multiply(self.f1, self.fother)(self.x)
        yother2 = np.multiply(self.f2, self.fother)(self.x)
        yother3 = np.multiply(self.f3, self.fother)(self.x)

        val = np.multiply(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_matmul(self):
        
        y1 = np.matmul(np.array([self.f1, 2, 3]), np.array([5, 2, 1]))(self.x)
        y2 = np.matmul(np.array([self.f2, 2, 3]), np.array([5, 2, 1]))(self.x)
        y3 = np.matmul(np.array([self.f3, 2, 3]), np.array([5, 2, 1]))(self.x)

        val = 5 * self.y + 7
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.matmul(self.f1, self.fother)(self.x)
        yother2 = np.matmul(self.f2, self.fother)(self.x)
        yother3 = np.matmul(self.f3, self.fother)(self.x)

        val = np.matmul(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_divide(self):
        
        y1 = np.divide(self.f1, 5)(self.x)
        y2 = np.divide(self.f2, 5)(self.x)
        y3 = np.divide(self.f3, 5)(self.x)

        val = np.divide(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.divide(self.f1, self.fother)(self.x)
        yother2 = np.divide(self.f2, self.fother)(self.x)
        yother3 = np.divide(self.f3, self.fother)(self.x)

        val = np.divide(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_logaddexp(self):
        
        y1 = np.logaddexp(self.f1, 5)(self.x)
        y2 = np.logaddexp(self.f2, 5)(self.x)
        y3 = np.logaddexp(self.f3, 5)(self.x)

        val = np.logaddexp(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.logaddexp(self.f1, self.fother)(self.x)
        yother2 = np.logaddexp(self.f2, self.fother)(self.x)
        yother3 = np.logaddexp(self.f3, self.fother)(self.x)

        val = np.logaddexp(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_logaddexp2(self):
        
        y1 = np.logaddexp2(self.f1, 5)(self.x)
        y2 = np.logaddexp2(self.f2, 5)(self.x)
        y3 = np.logaddexp2(self.f3, 5)(self.x)

        val = np.logaddexp2(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.logaddexp2(self.f1, self.fother)(self.x)
        yother2 = np.logaddexp2(self.f2, self.fother)(self.x)
        yother3 = np.logaddexp2(self.f3, self.fother)(self.x)

        val = np.logaddexp2(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_true_divide(self):
        
        y1 = np.true_divide(self.f1, 5)(self.x)
        y2 = np.true_divide(self.f2, 5)(self.x)
        y3 = np.true_divide(self.f3, 5)(self.x)

        val = np.true_divide(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.true_divide(self.f1, self.fother)(self.x)
        yother2 = np.true_divide(self.f2, self.fother)(self.x)
        yother3 = np.true_divide(self.f3, self.fother)(self.x)

        val = np.true_divide(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_floor_divide(self):
        
        y1 = np.floor_divide(self.f1, 5)(self.x)
        y2 = np.floor_divide(self.f2, 5)(self.x)
        y3 = np.floor_divide(self.f3, 5)(self.x)

        val = np.floor_divide(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.floor_divide(self.f1, self.fother)(self.x)
        yother2 = np.floor_divide(self.f2, self.fother)(self.x)
        yother3 = np.floor_divide(self.f3, self.fother)(self.x)

        val = np.floor_divide(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_negative(self):
        
        y1 = np.negative(self.f1)(self.x)
        y2 = np.negative(self.f2)(self.x)
        y3 = np.negative(self.f3)(self.x)

        val = np.negative(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_positive(self):
        
        y1 = np.positive(self.f1)(self.x)
        y2 = np.positive(self.f2)(self.x)
        y3 = np.positive(self.f3)(self.x)

        val = np.positive(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_power(self):
        
        y1 = np.power(self.f1, 5)(self.x)
        y2 = np.power(self.f2, 5)(self.x)
        y3 = np.power(self.f3, 5)(self.x)

        val = np.array([i ** 5 for i in self.y])
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.power(self.f1, self.fother)(self.x)
        yother2 = np.power(self.f2, self.fother)(self.x)
        yother3 = np.power(self.f3, self.fother)(self.x)

        val = []
        for i in range(self.y.size):
            if self.yother[i] % 2 == 0:
                val.append(abs(self.y[i]) ** self.yother[i])
            elif self.y[i] < 0:
                val.append(-(abs(self.y[i]) ** self.yother[i]))
            else:
                val.append(self.y[i] ** self.yother[i])

        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_float_power(self):
        
        y1 = np.float_power(self.f1, 5)(self.x)
        y2 = np.float_power(self.f2, 5)(self.x)
        y3 = np.float_power(self.f3, 5)(self.x)

        val = np.float_power(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.float_power(self.f1, self.fother)(self.x)
        yother2 = np.float_power(self.f2, self.fother)(self.x)
        yother3 = np.float_power(self.f3, self.fother)(self.x)

        val = []
        for i in range(self.y.size):
            if self.yother[i] % 2 == 0:
                val.append(abs(self.y[i]) ** self.yother[i])
            elif self.y[i] < 0:
                val.append(-(abs(self.y[i]) ** self.yother[i]))
            else:
                val.append(self.y[i] ** self.yother[i])

        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_remainder(self):
        
        y1 = np.remainder(self.f1, 5)(self.x)
        y2 = np.remainder(self.f2, 5)(self.x)
        y3 = np.remainder(self.f3, 5)(self.x)

        val = np.remainder(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.remainder(self.f1, self.fother)(self.x)
        yother2 = np.remainder(self.f2, self.fother)(self.x)
        yother3 = np.remainder(self.f3, self.fother)(self.x)

        val = np.remainder(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_mod(self):
        
        y1 = np.mod(self.f1, 5)(self.x)
        y2 = np.mod(self.f2, 5)(self.x)
        y3 = np.mod(self.f3, 5)(self.x)

        val = np.mod(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.mod(self.f1, self.fother)(self.x)
        yother2 = np.mod(self.f2, self.fother)(self.x)
        yother3 = np.mod(self.f3, self.fother)(self.x)

        val = np.mod(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_fmod(self):
        
        y1 = np.fmod(self.f1, 5)(self.x)
        y2 = np.fmod(self.f2, 5)(self.x)
        y3 = np.fmod(self.f3, 5)(self.x)

        val = np.fmod(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.fmod(self.f1, self.fother)(self.x)
        yother2 = np.fmod(self.f2, self.fother)(self.x)
        yother3 = np.fmod(self.f3, self.fother)(self.x)

        val = np.fmod(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_divmod(self):
        
        y1 = np.divmod(self.f1, 5)(self.x)
        y2 = np.divmod(self.f2, 5)(self.x)
        y3 = np.divmod(self.f3, 5)(self.x)

        val = np.divmod(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.divmod(self.f1, self.fother)(self.x)
        yother2 = np.divmod(self.f2, self.fother)(self.x)
        yother3 = np.divmod(self.f3, self.fother)(self.x)

        val = np.divmod(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))
    
    def test_absolute(self):

        y1 = np.absolute(self.f1)(self.x)
        y2 = np.absolute(self.f2)(self.x)
        y3 = np.absolute(self.f3)(self.x)

        val = np.absolute(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_fabs(self):
        
        y1 = np.fabs(self.f1)(self.x)
        y2 = np.fabs(self.f2)(self.x)
        y3 = np.fabs(self.f3)(self.x)

        val = np.fabs(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_rint(self):
        
        y1 = np.rint(self.f1)(self.x)
        y2 = np.rint(self.f2)(self.x)
        y3 = np.rint(self.f3)(self.x)

        val = np.rint(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_sign(self):
        
        y1 = np.sign(self.f1)(self.x)
        y2 = np.sign(self.f2)(self.x)
        y3 = np.sign(self.f3)(self.x)

        val = np.sign(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_heaviside(self):
        
        y1 = np.heaviside(self.f1, 5)(self.x)
        y2 = np.heaviside(self.f2, 5)(self.x)
        y3 = np.heaviside(self.f3, 5)(self.x)

        val = np.heaviside(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.heaviside(self.f1, self.fother)(self.x)
        yother2 = np.heaviside(self.f2, self.fother)(self.x)
        yother3 = np.heaviside(self.f3, self.fother)(self.x)

        val = np.heaviside(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_conj(self):
        
        y1 = np.conj(self.f1)(self.x)
        y2 = np.conj(self.f2)(self.x)
        y3 = np.conj(self.f3)(self.x)

        val = np.conj(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_conjugate(self):
        
        y1 = np.conjugate(self.f1)(self.x)
        y2 = np.conjugate(self.f2)(self.x)
        y3 = np.conjugate(self.f3)(self.x)

        val = np.conjugate(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_exp(self):
        
        y1 = np.exp(self.f1)(self.x)
        y2 = np.exp(self.f2)(self.x)
        y3 = np.exp(self.f3)(self.x)

        val = np.exp(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_exp2(self):
        
        y1 = np.exp2(self.f1)(self.x)
        y2 = np.exp2(self.f2)(self.x)
        y3 = np.exp2(self.f3)(self.x)

        val = np.exp2(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_log(self):
        
        y1 = np.log(self.f1)(self.x[50:])
        y2 = np.log(self.f2)(self.x[50:])
        y3 = np.log(self.f3)(self.x[50:])

        val = np.log(self.y[50:])
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_log2(self):
        
        y1 = np.log2(self.f1)(self.x[50:])
        y2 = np.log2(self.f2)(self.x[50:])
        y3 = np.log2(self.f3)(self.x[50:])

        val = np.log2(self.y[50:])
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_log10(self):
        
        y1 = np.log10(self.f1)(self.x[50:])
        y2 = np.log10(self.f2)(self.x[50:])
        y3 = np.log10(self.f3)(self.x[50:])

        val = np.log10(self.y[50:])
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_expm1(self):
        
        y1 = np.expm1(self.f1)(self.x)
        y2 = np.expm1(self.f2)(self.x)
        y3 = np.expm1(self.f3)(self.x)

        val = np.expm1(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_log1p(self):
        
        y1 = np.log1p(self.f1)(self.x[50:])
        y2 = np.log1p(self.f2)(self.x[50:])
        y3 = np.log1p(self.f3)(self.x[50:])

        val = np.log1p(self.y[50:])
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_sqrt(self):
        
        y1 = np.sqrt(self.f1)(self.x[50:])
        y2 = np.sqrt(self.f2)(self.x[50:])
        y3 = np.sqrt(self.f3)(self.x[50:])

        val = np.sqrt(self.y[50:])
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_square(self):
        
        y1 = np.square(self.f1)(self.x)
        y2 = np.square(self.f2)(self.x)
        y3 = np.square(self.f3)(self.x)

        val = np.square(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_cbrt(self):
        
        y1 = np.cbrt(self.f1)(self.x)
        y2 = np.cbrt(self.f2)(self.x)
        y3 = np.cbrt(self.f3)(self.x)

        val = np.cbrt(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_reciprocal(self):
        
        y1 = np.reciprocal(self.f1)(self.x)
        y2 = np.reciprocal(self.f2)(self.x)
        y3 = np.reciprocal(self.f3)(self.x)

        val = np.reciprocal(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))


class TestStrokeNumpyTrig(TestCase):

    def setUp(self):

        self.x = np.linspace(-1, 1, 10)
        self.y = np.exp(self.x) + np.cos(np.pi * self.x) - 1

        self.f1 = Stroke(self.x, self.y, "linear", "poly")
        self.f2 = Stroke(self.x, self.y, "quadratic", "poly")
        self.f3 = Stroke(self.x, self.y, "cubic", "poly")

        self.xother = np.linspace(-1, 1, 10)
        self.yother = np.sin(2 * self.xother)

        self.fother = Stroke(self.xother, self.yother, "linear", "poly")
    
    def test_sin(self):
        
        y1 = np.sin(self.f1)(self.x)
        y2 = np.sin(self.f2)(self.x)
        y3 = np.sin(self.f3)(self.x)

        val = np.sin(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_cos(self):
        
        y1 = np.cos(self.f1)(self.x)
        y2 = np.cos(self.f2)(self.x)
        y3 = np.cos(self.f3)(self.x)

        val = np.cos(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_tan(self):
        
        y1 = np.tan(self.f1)(self.x)
        y2 = np.tan(self.f2)(self.x)
        y3 = np.tan(self.f3)(self.x)

        val = np.tan(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arcsin(self):
        
        # Need to scale to be within arcsin domain.
        y1 = np.arcsin(self.f1 / 3 + 0.6)(self.x)
        y2 = np.arcsin(self.f2 / 3 + 0.6)(self.x)
        y3 = np.arcsin(self.f3 / 3 + 0.6)(self.x)

        val = np.arcsin(self.y / 3 + 0.6)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arccos(self):
        
        # Need to scale to be within arccos domain.
        y1 = np.arccos(self.f1 / 3 + 0.6)(self.x)
        y2 = np.arccos(self.f2 / 3 + 0.6)(self.x)
        y3 = np.arccos(self.f3 / 3 + 0.6)(self.x)

        val = np.arccos(self.y / 3 + 0.6)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arctan(self):
        
        y1 = np.arctan(self.f1)(self.x)
        y2 = np.arctan(self.f2)(self.x)
        y3 = np.arctan(self.f3)(self.x)

        val = np.arctan(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arctan2(self):
        
        y1 = np.arctan2(self.f1, 3)(self.x)
        y2 = np.arctan2(self.f2, 3)(self.x)
        y3 = np.arctan2(self.f3, 3)(self.x)

        val = np.arctan2(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.arctan2(self.f1, self.fother)(self.x)
        yother2 = np.arctan2(self.f2, self.fother)(self.x)
        yother3 = np.arctan2(self.f3, self.fother)(self.x)

        val = np.arctan2(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_hypot(self):
        
        y1 = np.hypot(self.f1, 5)(self.x)
        y2 = np.hypot(self.f2, 5)(self.x)
        y3 = np.hypot(self.f3, 5)(self.x)

        val = np.hypot(self.y, 5)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.hypot(self.f1, self.fother)(self.x)
        yother2 = np.hypot(self.f2, self.fother)(self.x)
        yother3 = np.hypot(self.f3, self.fother)(self.x)

        val = np.hypot(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_sinh(self):
        
        y1 = np.sinh(self.f1)(self.x)
        y2 = np.sinh(self.f2)(self.x)
        y3 = np.sinh(self.f3)(self.x)

        val = np.sinh(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_cosh(self):
        
        y1 = np.cosh(self.f1)(self.x)
        y2 = np.cosh(self.f2)(self.x)
        y3 = np.cosh(self.f3)(self.x)

        val = np.cosh(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_tanh(self):
        
        y1 = np.tanh(self.f1)(self.x)
        y2 = np.tanh(self.f2)(self.x)
        y3 = np.tanh(self.f3)(self.x)

        val = np.tanh(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arcsinh(self):
        
        y1 = np.arcsinh(self.f1)(self.x)
        y2 = np.arcsinh(self.f2)(self.x)
        y3 = np.arcsinh(self.f3)(self.x)

        val = np.arcsinh(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arccosh(self):
        
        # Need to scale to be within arccosh domain.
        y1 = np.arccosh(self.f1 + 3)(self.x)
        y2 = np.arccosh(self.f2 + 3)(self.x)
        y3 = np.arccosh(self.f3 + 3)(self.x)

        val = np.arccosh(self.y + 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_arctanh(self):
        
        # Need to scale to be within arctanh domain.
        y1 = np.arctanh(self.f1 / 3)(self.x)
        y2 = np.arctanh(self.f2 / 3)(self.x)
        y3 = np.arctanh(self.f3 / 3)(self.x)

        val = np.arctanh(self.y / 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_degrees(self):
        
        y1 = np.degrees(self.f1)(self.x)
        y2 = np.degrees(self.f2)(self.x)
        y3 = np.degrees(self.f3)(self.x)

        val = np.degrees(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_radians(self):
        
        y1 = np.radians(self.f1)(self.x)
        y2 = np.radians(self.f2)(self.x)
        y3 = np.radians(self.f3)(self.x)

        val = np.radians(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_deg2rad(self):
        
        y1 = np.deg2rad(self.f1)(self.x)
        y2 = np.deg2rad(self.f2)(self.x)
        y3 = np.deg2rad(self.f3)(self.x)

        val = np.deg2rad(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_rad2deg(self):
        
        y1 = np.rad2deg(self.f1)(self.x)
        y2 = np.rad2deg(self.f2)(self.x)
        y3 = np.rad2deg(self.f3)(self.x)

        val = np.rad2deg(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))


class TestStrokeNumpyComparison(TestCase):

    def setUp(self):

        self.x = np.linspace(-1, 1, 10)
        self.y = np.exp(self.x) + np.cos(np.pi * self.x) - 1

        self.f1 = Stroke(self.x, self.y, "linear", "poly")
        self.f2 = Stroke(self.x, self.y, "quadratic", "poly")
        self.f3 = Stroke(self.x, self.y, "cubic", "poly")

        self.xother = np.linspace(-1, 1, 10)
        self.yother = np.sin(2 * self.xother)

        self.fother = Stroke(self.xother, self.yother, "linear", "poly")
    
    def test_greater(self):
        
        y1 = np.greater(self.f1, 3)(self.x)
        y2 = np.greater(self.f2, 3)(self.x)
        y3 = np.greater(self.f3, 3)(self.x)

        val = np.greater(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.greater(3, self.f1)(self.x)
        y2 = np.greater(3, self.f2)(self.x)
        y3 = np.greater(3, self.f3)(self.x)

        val = np.greater(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.greater(self.f1, self.fother)(self.x)
        yother2 = np.greater(self.f2, self.fother)(self.x)
        yother3 = np.greater(self.f3, self.fother)(self.x)

        val = np.greater(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_greater_equal(self):
        
        y1 = np.greater_equal(self.f1, 3)(self.x)
        y2 = np.greater_equal(self.f2, 3)(self.x)
        y3 = np.greater_equal(self.f3, 3)(self.x)

        val = np.greater_equal(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.greater_equal(3, self.f1)(self.x)
        y2 = np.greater_equal(3, self.f2)(self.x)
        y3 = np.greater_equal(3, self.f3)(self.x)

        val = np.greater_equal(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.greater_equal(self.f1, self.fother)(self.x)
        yother2 = np.greater_equal(self.f2, self.fother)(self.x)
        yother3 = np.greater_equal(self.f3, self.fother)(self.x)

        val = np.greater_equal(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_less(self):
        
        y1 = np.less(self.f1, 3)(self.x)
        y2 = np.less(self.f2, 3)(self.x)
        y3 = np.less(self.f3, 3)(self.x)

        val = np.less(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.less(3, self.f1)(self.x)
        y2 = np.less(3, self.f2)(self.x)
        y3 = np.less(3, self.f3)(self.x)

        val = np.less(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.less(self.f1, self.fother)(self.x)
        yother2 = np.less(self.f2, self.fother)(self.x)
        yother3 = np.less(self.f3, self.fother)(self.x)

        val = np.less(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_less_equal(self):
        
        y1 = np.less_equal(self.f1, 3)(self.x)
        y2 = np.less_equal(self.f2, 3)(self.x)
        y3 = np.less_equal(self.f3, 3)(self.x)

        val = np.less_equal(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.less_equal(3, self.f1)(self.x)
        y2 = np.less_equal(3, self.f2)(self.x)
        y3 = np.less_equal(3, self.f3)(self.x)

        val = np.less_equal(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.less_equal(self.f1, self.fother)(self.x)
        yother2 = np.less_equal(self.f2, self.fother)(self.x)
        yother3 = np.less_equal(self.f3, self.fother)(self.x)

        val = np.less_equal(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_not_equal(self):
        
        y1 = np.not_equal(self.f1, 3)(self.x)
        y2 = np.not_equal(self.f2, 3)(self.x)
        y3 = np.not_equal(self.f3, 3)(self.x)

        val = np.not_equal(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.not_equal(3, self.f1)(self.x)
        y2 = np.not_equal(3, self.f2)(self.x)
        y3 = np.not_equal(3, self.f3)(self.x)

        val = np.not_equal(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.not_equal(self.f1, self.fother)(self.x)
        yother2 = np.not_equal(self.f2, self.fother)(self.x)
        yother3 = np.not_equal(self.f3, self.fother)(self.x)

        val = np.not_equal(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_equal(self):
        
        y1 = np.equal(self.f1, 3)(self.x)
        y2 = np.equal(self.f2, 3)(self.x)
        y3 = np.equal(self.f3, 3)(self.x)

        val = np.equal(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.equal(3, self.f1)(self.x)
        y2 = np.equal(3, self.f2)(self.x)
        y3 = np.equal(3, self.f3)(self.x)

        val = np.equal(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.equal(self.f1, self.fother)(self.x)
        yother2 = np.equal(self.f2, self.fother)(self.x)
        yother3 = np.equal(self.f3, self.fother)(self.x)

        val = np.equal(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_logical_and(self):
        
        y1 = np.logical_and(self.f1, 3)(self.x)
        y2 = np.logical_and(self.f2, 3)(self.x)
        y3 = np.logical_and(self.f3, 3)(self.x)

        val = np.logical_and(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.logical_and(3, self.f1)(self.x)
        y2 = np.logical_and(3, self.f2)(self.x)
        y3 = np.logical_and(3, self.f3)(self.x)

        val = np.logical_and(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.logical_and(self.f1, self.fother)(self.x)
        yother2 = np.logical_and(self.f2, self.fother)(self.x)
        yother3 = np.logical_and(self.f3, self.fother)(self.x)

        val = np.logical_and(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_logical_or(self):
        
        y1 = np.logical_or(self.f1, 3)(self.x)
        y2 = np.logical_or(self.f2, 3)(self.x)
        y3 = np.logical_or(self.f3, 3)(self.x)

        val = np.logical_or(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.logical_or(3, self.f1)(self.x)
        y2 = np.logical_or(3, self.f2)(self.x)
        y3 = np.logical_or(3, self.f3)(self.x)

        val = np.logical_or(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.logical_or(self.f1, self.fother)(self.x)
        yother2 = np.logical_or(self.f2, self.fother)(self.x)
        yother3 = np.logical_or(self.f3, self.fother)(self.x)

        val = np.logical_or(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_logical_xor(self):
        
        y1 = np.logical_xor(self.f1, 3)(self.x)
        y2 = np.logical_xor(self.f2, 3)(self.x)
        y3 = np.logical_xor(self.f3, 3)(self.x)

        val = np.logical_xor(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.logical_xor(3, self.f1)(self.x)
        y2 = np.logical_xor(3, self.f2)(self.x)
        y3 = np.logical_xor(3, self.f3)(self.x)

        val = np.logical_xor(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.logical_xor(self.f1, self.fother)(self.x)
        yother2 = np.logical_xor(self.f2, self.fother)(self.x)
        yother3 = np.logical_xor(self.f3, self.fother)(self.x)

        val = np.logical_xor(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_logical_not(self):
        
        y1 = np.logical_not(self.f1)(self.x)
        y2 = np.logical_not(self.f2)(self.x)
        y3 = np.logical_not(self.f3)(self.x)

        val = np.logical_not(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_maximum(self):
        
        y1 = np.maximum(self.f1, 3)(self.x)
        y2 = np.maximum(self.f2, 3)(self.x)
        y3 = np.maximum(self.f3, 3)(self.x)

        val = np.maximum(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.maximum(3, self.f1)(self.x)
        y2 = np.maximum(3, self.f2)(self.x)
        y3 = np.maximum(3, self.f3)(self.x)

        val = np.maximum(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.maximum(self.f1, self.fother)(self.x)
        yother2 = np.maximum(self.f2, self.fother)(self.x)
        yother3 = np.maximum(self.f3, self.fother)(self.x)

        val = np.maximum(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_minimum(self):
        
        y1 = np.minimum(self.f1, 3)(self.x)
        y2 = np.minimum(self.f2, 3)(self.x)
        y3 = np.minimum(self.f3, 3)(self.x)

        val = np.minimum(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.minimum(3, self.f1)(self.x)
        y2 = np.minimum(3, self.f2)(self.x)
        y3 = np.minimum(3, self.f3)(self.x)

        val = np.minimum(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.minimum(self.f1, self.fother)(self.x)
        yother2 = np.minimum(self.f2, self.fother)(self.x)
        yother3 = np.minimum(self.f3, self.fother)(self.x)

        val = np.minimum(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_fmax(self):
        
        y1 = np.fmax(self.f1, 3)(self.x)
        y2 = np.fmax(self.f2, 3)(self.x)
        y3 = np.fmax(self.f3, 3)(self.x)

        val = np.fmax(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.fmax(3, self.f1)(self.x)
        y2 = np.fmax(3, self.f2)(self.x)
        y3 = np.fmax(3, self.f3)(self.x)

        val = np.fmax(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.fmax(self.f1, self.fother)(self.x)
        yother2 = np.fmax(self.f2, self.fother)(self.x)
        yother3 = np.fmax(self.f3, self.fother)(self.x)

        val = np.fmax(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_fmin(self):
        
        y1 = np.fmin(self.f1, 3)(self.x)
        y2 = np.fmin(self.f2, 3)(self.x)
        y3 = np.fmin(self.f3, 3)(self.x)

        val = np.fmin(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.fmin(3, self.f1)(self.x)
        y2 = np.fmin(3, self.f2)(self.x)
        y3 = np.fmin(3, self.f3)(self.x)

        val = np.fmin(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.fmin(self.f1, self.fother)(self.x)
        yother2 = np.fmin(self.f2, self.fother)(self.x)
        yother3 = np.fmin(self.f3, self.fother)(self.x)

        val = np.fmin(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))


class TestStrokeNumpyFloating(TestCase):

    def setUp(self):

        self.x = np.linspace(-1, 1, 10)
        self.y = np.exp(self.x) + np.cos(np.pi * self.x) - 1

        self.f1 = Stroke(self.x, self.y, "linear", "poly")
        self.f2 = Stroke(self.x, self.y, "quadratic", "poly")
        self.f3 = Stroke(self.x, self.y, "cubic", "poly")

        self.xother = np.linspace(-1, 1, 10)
        self.yother = np.sin(2 * self.xother)

        self.fother = Stroke(self.xother, self.yother, "linear", "poly")
    
    def test_isfinite(self):
        
        y1 = np.isfinite(self.f1)(self.x)
        y2 = np.isfinite(self.f2)(self.x)
        y3 = np.isfinite(self.f3)(self.x)

        val = np.isfinite(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_isinf(self):
        
        y1 = np.isinf(self.f1)(self.x)
        y2 = np.isinf(self.f2)(self.x)
        y3 = np.isinf(self.f3)(self.x)

        val = np.isinf(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_isnan(self):
        
        y1 = np.isnan(self.f1)(self.x)
        y2 = np.isnan(self.f2)(self.x)
        y3 = np.isnan(self.f3)(self.x)

        val = np.isnan(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_fabs(self):
        
        y1 = np.fabs(self.f1)(self.x)
        y2 = np.fabs(self.f2)(self.x)
        y3 = np.fabs(self.f3)(self.x)

        val = np.fabs(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_signbit(self):
        
        y1 = np.signbit(self.f1)(self.x)
        y2 = np.signbit(self.f2)(self.x)
        y3 = np.signbit(self.f3)(self.x)

        val = np.signbit(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_copysign(self):
        
        y1 = np.copysign(self.f1, 3)(self.x)
        y2 = np.copysign(self.f2, 3)(self.x)
        y3 = np.copysign(self.f3, 3)(self.x)

        val = np.copysign(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.copysign(3, self.f1)(self.x)
        y2 = np.copysign(3, self.f2)(self.x)
        y3 = np.copysign(3, self.f3)(self.x)

        val = np.copysign(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.copysign(self.f1, self.fother)(self.x)
        yother2 = np.copysign(self.f2, self.fother)(self.x)
        yother3 = np.copysign(self.f3, self.fother)(self.x)

        val = np.copysign(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_nextafter(self):
        
        y1 = np.nextafter(self.f1, 3)(self.x)
        y2 = np.nextafter(self.f2, 3)(self.x)
        y3 = np.nextafter(self.f3, 3)(self.x)

        val = np.nextafter(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.nextafter(3, self.f1)(self.x)
        y2 = np.nextafter(3, self.f2)(self.x)
        y3 = np.nextafter(3, self.f3)(self.x)

        val = np.nextafter(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.nextafter(self.f1, self.fother)(self.x)
        yother2 = np.nextafter(self.f2, self.fother)(self.x)
        yother3 = np.nextafter(self.f3, self.fother)(self.x)

        val = np.nextafter(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_spacing(self):
        
        y1 = np.spacing(self.f1)(self.x)
        y2 = np.spacing(self.f2)(self.x)
        y3 = np.spacing(self.f3)(self.x)

        val = np.spacing(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_modf(self):
        
        y1 = np.modf(self.f1)(self.x)
        y2 = np.modf(self.f2)(self.x)
        y3 = np.modf(self.f3)(self.x)

        val = np.modf(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_frexp(self):
        
        y1 = np.frexp(self.f1)(self.x)
        y2 = np.frexp(self.f2)(self.x)
        y3 = np.frexp(self.f3)(self.x)

        val = np.frexp(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_fmod(self):
        
        y1 = np.fmod(self.f1, 3)(self.x)
        y2 = np.fmod(self.f2, 3)(self.x)
        y3 = np.fmod(self.f3, 3)(self.x)

        val = np.fmod(self.y, 3)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        y1 = np.fmod(3, self.f1)(self.x)
        y2 = np.fmod(3, self.f2)(self.x)
        y3 = np.fmod(3, self.f3)(self.x)

        val = np.fmod(3, self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

        yother1 = np.fmod(self.f1, self.fother)(self.x)
        yother2 = np.fmod(self.f2, self.fother)(self.x)
        yother3 = np.fmod(self.f3, self.fother)(self.x)

        val = np.fmod(self.y, self.yother)
        self.assertTrue(np.allclose(yother1, val, rtol=0.01))
        self.assertTrue(np.allclose(yother2, val, rtol=0.01))
        self.assertTrue(np.allclose(yother3, val, rtol=0.01))

    def test_floor(self):
        
        y1 = np.floor(self.f1)(self.x)
        y2 = np.floor(self.f2)(self.x)
        y3 = np.floor(self.f3)(self.x)

        val = np.floor(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_ceil(self):
        
        y1 = np.ceil(self.f1)(self.x)
        y2 = np.ceil(self.f2)(self.x)
        y3 = np.ceil(self.f3)(self.x)

        val = np.ceil(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))

    def test_trunc(self):
        
        y1 = np.trunc(self.f1)(self.x)
        y2 = np.trunc(self.f2)(self.x)
        y3 = np.trunc(self.f3)(self.x)

        val = np.trunc(self.y)
        self.assertTrue(np.allclose(y1, val, rtol=0.01))
        self.assertTrue(np.allclose(y2, val, rtol=0.01))
        self.assertTrue(np.allclose(y3, val, rtol=0.01))
