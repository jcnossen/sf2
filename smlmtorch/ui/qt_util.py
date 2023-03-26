# -*- coding: utf-8 -*-
#author: jelmer cnossen 2021/2022
#license: public domain
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5 import QtCore
import sys

def error_box(txt, title='Error'):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText(txt)
    msg.setWindowTitle(title)
    msg.exec_()

class NeedsQt:
    def __init__(self):
        self.app = QApplication.instance()
        self.appOwner = self.app is None
        if self.appOwner:
            self.app = QApplication(sys.argv)

    def close(self):
        if self.appOwner and self.app is not None:
            del self.app
            
    def __del__(self):
        self.close()
      
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()        

def needs_qt(func):
    """
    Decorator to make sure QApplication is only instantiated once
    """
    def run(*args,**kwargs):
        app = QApplication.instance()
        appOwner = app is None
        if appOwner:
            app = QApplication(sys.argv)
        
        r = func(*args,**kwargs)
        
        if appOwner:
            del app
            
        return r
    return run

def force_top(window):
    # bring window to top and act like a "normal" window!
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # set always on top flag, makes window disappear
    window.show() # makes window reappear, but it's ALWAYS on top
    window.setWindowFlags(window.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint) # clear always on top flag, makes window disappear
    window.show() # makes window reappear, acts like normal window now (on top now but can be underneath if you raise another window)