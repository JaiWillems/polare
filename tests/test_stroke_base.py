

from polare import Stroke
from unittest import TestCase
import numpy as np


class TestStrokeBase(TestCase):

    def setUp(self):

        self.x = np.linspace(-1, 1, 10)
        self.y = np.exp(self.x) + np.cos(np.pi * self.x) - 1

        self.f1 = Stroke(self.x, self.y, "linear")
        self.f2 = Stroke(self.x, self.y, "quadratic")
        self.f3 = Stroke(self.x, self.y, "cubic")

        self.xnew = np.linspace(-1, 1, 100)
        self.ynew = np.exp(self.xnew) + np.cos(np.pi * self.xnew) - 1

        self.xother = np.linspace(-1, 1, 10)
        self.yother = np.sin(self.xother)

        self.fother = Stroke(self.xother, self.yother)

        self.xothernew = np.linspace(-1, 1, 100)
        self.yothernew = np.sin(self.xothernew)
    
    def test_pos(self):
        
        y1 = (+self.f1)(self.xnew)
        y2 = (+self.f2)(self.xnew)
        y3 = (+self.f3)(self.xnew)

        self.assertTrue(np.allclose(self.ynew, y1, atol=0.1))
        self.assertTrue(np.allclose(self.ynew, y2, atol=0.1))
        self.assertTrue(np.allclose(self.ynew, y3, atol=0.1))

    def test_neg(self):
        
        y1 = (-self.f1)(self.xnew)
        y2 = (-self.f2)(self.xnew)
        y3 = (-self.f3)(self.xnew)

        self.assertTrue(np.allclose(-self.ynew, y1, atol=0.1))
        self.assertTrue(np.allclose(-self.ynew, y2, atol=0.1))
        self.assertTrue(np.allclose(-self.ynew, y3, atol=0.1))
    
    def test_add(self):
        
        y1 = (self.f1 + 5)(self.xnew)
        y2 = (self.f2 + 5)(self.xnew)
        y3 = (self.f3 + 5)(self.xnew)

        self.assertTrue(np.allclose(self.ynew + 5, y1, atol=0.1))
        self.assertTrue(np.allclose(self.ynew + 5, y2, atol=0.1))
        self.assertTrue(np.allclose(self.ynew + 5, y3, atol=0.1))

        yother = (self.f1 + self.fother)(self.xnew)
        self.assertTrue(np.allclose(self.ynew + self.yothernew, yother, atol=0.1))
    
    def test_radd(self):

        y1 = (5 + self.f1)(self.xnew)
        y2 = (5 + self.f2)(self.xnew)
        y3 = (5 + self.f3)(self.xnew)

        self.assertTrue(np.allclose(5 + self.ynew, y1, atol=0.1))
        self.assertTrue(np.allclose(5 + self.ynew, y2, atol=0.1))
        self.assertTrue(np.allclose(5 + self.ynew, y3, atol=0.1))

        yother = (self.fother + self.f1)(self.xnew)
        self.assertTrue(np.allclose(self.yothernew + self.ynew, yother, atol=0.1))
    
    def test_sub(self):

        y1 = (self.f1 - 5)(self.xnew)
        y2 = (self.f2 - 5)(self.xnew)
        y3 = (self.f3 - 5)(self.xnew)

        self.assertTrue(np.allclose(self.ynew - 5, y1, atol=0.1))
        self.assertTrue(np.allclose(self.ynew - 5, y2, atol=0.1))
        self.assertTrue(np.allclose(self.ynew - 5, y3, atol=0.1))

        yother = (self.f1 - self.fother)(self.xnew)
        self.assertTrue(np.allclose(self.ynew - self.yothernew, yother, atol=0.1))
    
    def test_rsub(self):

        y1 = (5 - self.f1)(self.xnew)
        y2 = (5 - self.f2)(self.xnew)
        y3 = (5 - self.f3)(self.xnew)

        self.assertTrue(np.allclose(5 - self.ynew, y1, atol=0.1))
        self.assertTrue(np.allclose(5 - self.ynew, y2, atol=0.1))
        self.assertTrue(np.allclose(5 - self.ynew, y3, atol=0.1))

        yother = (self.fother - self.f1)(self.xnew)
        self.assertTrue(np.allclose(self.yothernew - self.ynew, yother, atol=0.1))
    
    def test_mul(self):
        
        y1 = (self.f1 * 5)(self.xnew)
        y2 = (self.f2 * 5)(self.xnew)
        y3 = (self.f3 * 5)(self.xnew)

        self.assertTrue(np.allclose(self.ynew * 5, y1, atol=0.4))
        self.assertTrue(np.allclose(self.ynew * 5, y2, atol=0.2))
        self.assertTrue(np.allclose(self.ynew * 5, y3, atol=0.05))

        yother = (self.f1 * self.fother)(self.xnew)
        self.assertTrue(np.allclose(self.ynew * self.yothernew, yother, atol=0.1))
    
    def test_rmul(self):

        y1 = (5 * self.f1)(self.xnew)
        y2 = (5 * self.f2)(self.xnew)
        y3 = (5 * self.f3)(self.xnew)

        self.assertTrue(np.allclose(5 * self.ynew, y1, atol=0.4))
        self.assertTrue(np.allclose(5 * self.ynew, y2, atol=0.2))
        self.assertTrue(np.allclose(5 * self.ynew, y3, atol=0.05))

        yother = (self.fother * self.f1)(self.xnew)
        self.assertTrue(np.allclose(self.yothernew * self.ynew, yother, atol=0.1))
    
    def test_truediv(self):
        
        y1 = (self.f1 / 5)(self.xnew)
        y2 = (self.f2 / 5)(self.xnew)
        y3 = (self.f3 / 5)(self.xnew)

        self.assertTrue(np.allclose(self.ynew / 5, y1, atol=0.05))
        self.assertTrue(np.allclose(self.ynew / 5, y2, atol=0.05))
        self.assertTrue(np.allclose(self.ynew / 5, y3, atol=0.05))

        yother = (self.f1 / self.fother)(self.xnew)
        self.assertTrue(np.allclose(self.ynew / self.yothernew, yother, atol=5.5))
    
    def test_rtruediv(self):

        y1 = (5 / self.f1)(self.xnew[50:])
        y2 = (5 / self.f2)(self.xnew[50:])
        y3 = (5 / self.f3)(self.xnew[50:])

        self.assertTrue(np.allclose(5 / self.ynew[50:], y1, atol=1.5))
        self.assertTrue(np.allclose(5 / self.ynew[50:], y2, atol=1.5))
        self.assertTrue(np.allclose(5 / self.ynew[50:], y3, atol=1.5))

        yother = (self.fother / self.f1)(self.xnew[50:])
        self.assertTrue(np.allclose(self.yothernew[50:] / self.ynew[50:], yother, atol=0.3))
    
    def test_pow(self):
        
        y1 = (self.f1 ** 5)(self.xnew)
        y2 = (self.f2 ** 5)(self.xnew)
        y3 = (self.f3 ** 5)(self.xnew)

        self.assertTrue(np.allclose(self.ynew ** 5, y1, atol=2))
        self.assertTrue(np.allclose(self.ynew ** 5, y2, atol=2))
        self.assertTrue(np.allclose(self.ynew ** 5, y3, atol=2))

        yother = (abs(self.f1) ** abs(self.fother))(self.xnew)
        self.assertTrue(np.allclose(abs(self.ynew) ** abs(self.yothernew), yother, atol=2))
    
    def test_rpow(self):

        y1 = (5 ** self.f1)(self.xnew)
        y2 = (5 ** self.f2)(self.xnew)
        y3 = (5 ** self.f3)(self.xnew)

        self.assertTrue(np.allclose(5 ** self.ynew, y1, atol=2))
        self.assertTrue(np.allclose(5 ** self.ynew, y2, atol=2))
        self.assertTrue(np.allclose(5 ** self.ynew, y3, atol=2))

        yother = (abs(self.fother) ** abs(self.f1))(self.xnew)
        self.assertTrue(np.allclose(abs(self.yothernew) ** abs(self.ynew), yother, atol=2))

    def test_abs(self):

        y1 = (abs(self.f1))(self.xnew)
        y2 = (abs(self.f2))(self.xnew)
        y3 = (abs(self.f3))(self.xnew)

        self.assertTrue(np.allclose(abs(self.ynew), y1, atol=0.1))
        self.assertTrue(np.allclose(abs(self.ynew), y2, atol=0.1))
        self.assertTrue(np.allclose(abs(self.ynew), y3, atol=0.1))
