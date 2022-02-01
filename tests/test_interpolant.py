

from polare.interpolant import Interp
from unittest import TestCase
import numpy as np


class TestPolyInterp(TestCase):

    def setUp(self):
        
        self.x = np.linspace(0, 10, 100)
        self.y = np.exp(self.x) + np.cos(np.pi * self.x) + 1

        self.xnew = np.linspace(0, 10, 1000)
        self.ynew = np.exp(self.xnew) + np.cos(np.pi * self.xnew) + 1
    
    def test_linear(self):

        ytest = Interp(self.x, self.y, "linear")(self.xnew)
        self.assertTrue(np.allclose(self.ynew, ytest, rtol=0.01))
    
    def test_quadratic(self):

        ytest = Interp(self.x, self.y, "quadratic")(self.xnew)
        self.assertTrue(np.allclose(self.ynew, ytest, rtol=0.01))
    
    def test_cubic(self):

        ytest = Interp(self.x, self.y, "cubic")(self.xnew)
        self.assertTrue(np.allclose(self.ynew, ytest, rtol=0.01))
    