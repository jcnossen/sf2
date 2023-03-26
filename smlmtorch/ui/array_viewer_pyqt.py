"""
super light n-d array viewer (like napari) to help debugging
author: jelmer cnossen 2021/2022
license: public domain
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider,QMenuBar
from PyQt5 import QtWidgets
import pyqtgraph as pg
from smlmtorch.ui.qt_util import needs_qt

import numpy as np


#class ImageViewWindow(QtWidgets.QMainWindow):
class TraceViewWindow(QtWidgets.QDialog):
    def __init__(self, traces, trace_labels=None, callback=None, *args, **kwargs):
        super(TraceViewWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle('Trace Viewer')
        
        if len(traces.shape) == 1:
            traces = traces[:,None]
            
        if traces.shape[1] > 1 and trace_labels is None:
            trace_labels = [f'{i}' for i in range(traces.shape[1])]
        
        self.menu = QMenuBar()
        ltop = QtWidgets.QHBoxLayout()
       
        layout = QtWidgets.QVBoxLayout()
        ltop.addLayout(layout)
        ltop.setMenuBar(self.menu)

        ctlLayout = QtWidgets.QGridLayout()

        self.info = QtWidgets.QLabel()
        ctlLayout.addWidget(self.info, 0, 2, Qt.AlignLeft)

        layout.addLayout(ctlLayout)
        
        view = pg.PlotWidget()
        layout.addWidget(view)
        self.plotItem = view.plotItem

        self.setLayout(ltop)        
        self.plotItem.addLegend()

        for i in range(traces.shape[1]):
            pen = pg.mkPen((i,traces.shape[1]))
            self.plotItem.plot(traces[:,i], pen=pen, symbol='o', name=
                               trace_labels[i] if trace_labels is not None else None)
            
        if callback is not None:
            callback(self.plotItem)

    def update(self):
        ...

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        #self.closeCamera()
        ...

        
    def closeEvent(self, event):
        event.accept()



    
#class ImageViewWindow(QtWidgets.QMainWindow):
class ImageViewWindow(QtWidgets.QDialog):
    def __init__(self, img, *args, **kwargs):
        super(ImageViewWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle('Image Viewer')
                
        
        self.menu=QMenuBar()
        ltop = QtWidgets.QHBoxLayout()
       
        layout = QtWidgets.QVBoxLayout()
        ltop.addLayout(layout)
        ltop.setMenuBar(self.menu)
        #btnstart = QtWidgets.QPushButton("Take ZStack")
        #btnstart.setMaximumSize(100,32)
        #btnstart.clicked.connect(self.recordZStack)

        ctlLayout = QtWidgets.QGridLayout()
        #ctlLayout = QtWidgets.QHBoxLayout()

        sliders=[]        
        for i in range( len(img.shape)-2 ):
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(10)
            slider.setSingleStep(1)
            slider.setMaximum(img.shape[i]-1)
            slider.valueChanged.connect(self.sliderChange)
            layout.addWidget(slider)#, 0, 0)
            sliders.append(slider)
        self.sliders=sliders

        #layout.addWidget(btnstart)#, 0, 0)

        self.info = QtWidgets.QLabel()
        ctlLayout.addWidget(self.info, 0, 2, Qt.AlignLeft)

        layout.addLayout(ctlLayout)
        view = pg.ImageView()
        layout.addWidget(view)
        self.imv = view.imageItem
        #w = pg.GraphicsLayoutWidget()
        #layout.addWidget(w)
        
        #v = w.addViewBox(row=0, col=0)
        #self.imv = ImageItem()
        #v.addItem(self.imv)

        self.setLayout(ltop)
        
        #v = w.addViewBox(row=0, col=1)
        #self.zstackImv = ImageItem()
        #v.addItem(self.zstackImv)

        self.imv.setImage(img)
        self.data = img
        
        """
        h = img.shape[-2]
        w = img.shape[-1]
        
        if h > 600 or w > 600:
            scale = 600 / max(w,h)
            w *= scale
            h *= scale
            
        #if h < 50 and w < 
            
        x_space = self.width() - v.width()
        y_space = self.height() - v.height()
        
        self.resize(x_space+w, y_space+h)
        """
        
        self.sliderChange()

    def sliderChange(self):
        d = self.data
        ix=[]
        for s in self.sliders:
            ix.append(s.value())
            d=d[s.value()]
        self.imv.setImage(d.T)
        self.info.setText("["+','.join([str(i) for i in ix])+']')

    def update(self):
        ...

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        #self.closeCamera()
        ...

        
    def closeEvent(self, event):
        #print('close event')
        #self.closeCamera()
        event.accept()

@needs_qt
def trace_view(traces, trace_labels=None, modal=True, parent=None, title=None):
    if getattr(traces, "cpu", None): # convert torch tensors without importing torch
        traces = traces.detach().cpu().numpy()
    
    w = TraceViewWindow(np.array(traces), trace_labels=trace_labels, parent=parent)
    w.setModal(modal)

    app = QtWidgets.QApplication.instance()
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    if title is not None:
        w.setWindowTitle(title)
    if modal:
        w.exec_()
    else:
        w.show()
    return w


@needs_qt
def array_view(img, modal=True, parent=None, title=None):
    if getattr(img, "cpu", None): # convert torch tensors without importing torch
        img = img.detach().cpu().numpy()
    w = ImageViewWindow(np.array(img), parent=parent)
    w.setModal(modal)

    app = QtWidgets.QApplication.instance()
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    if title is not None:
        w.setWindowTitle(title)
    if modal:
        w.exec_()
    else:
        w.show()
    return w



if __name__ == '__main__':
    traces = np.random.normal(0,1,size=(500,3)).cumsum(0)
    trace_view(traces, title='Random traces')

    img = np.random.uniform(0, 100, size=(20,5,200,200)).astype(np.uint8)
    array_view(img, title='Random image')

    