from sympy import *
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    
def between(a, b, x):
    ''' determine if a point x is between a and b. a may be greater or less than b '''
    if a > b:
        return b <= x <= a
    if b > a:
        return a <= x <= b
    
def near(pt, alist, dist=15):
    for a in alist:
        x, y = a.ao.get_position()  #(bot left, bot right) data coords, not relative
        x = x - 5  
        y = y + 2.5
        if 0 < (pt[0] - x) < 25 and 0 < (y - pt[1]) < 25:
            return(True, a)
    return(False,None)

def inboxes(pt, boxlist):
    ''' returns true if pt is within one of the boxes in boxlist '''
    #with out:
    #    print(f" inboxes:{boxlist}, {pt}")
    for b in boxlist:
        if b.inbox(pt):
            return(True, b)
    return(False, None)


class avalue():
    ''' one of the values on the figure that can be filled in '''
    def __init__(self, value, pt, cl):
        self.value = value
        self.cl = cl   # color
        self.pt = pt   # point
    
    def add_anote(self, ax):
        self.ax = ax
        self.ao = self.ax.annotate("?", self.pt, c=self.cl, fontsize='x-small')

class astring():
    ''' a string that can be set visible or invisible '''
    def __init__(self, ax, string, pt, cl):
        self.string = string
        self.cl = cl   # color
        self.pt = pt   # point
        self.ax = ax
        self.ao = self.ax.annotate(self.string, self.pt, c="white", fontsize='x-small')
    
    def astring_visible(self):
        self.ao.set_color(self.cl)

    def astring_invisible(self):
        self.ao.set_color("white")


class abox():
    ''' one of the boxes in the graph that has a value '''
    def __init__(self, ax, value, left, bottom, right, top, anpt, cl, adj_anote_obj):
        self.ax = ax
        self.value = value  # correct value for annotation
        self.left = left
        self.right = right 
        self.bottom = bottom
        self.top = top
        self.anpt= anpt # x,y where expression should be listed
        self.cl = cl
        self.ao = self.ax.annotate("?", self.anpt, c=self.cl, fontsize='x-small')
        self.astr = adj_anote_obj   # 2ndary text for marking edges or none
            
    def inbox(self, pt):
        ''' true if point is within the box '''
        #with out:   #debug
        #    print(f" b.inbox: {pt}")
        x, y = pt  
        isbetween =  between(self.top, self.bottom, y) and between(self.left, self.right, x)
        return isbetween
    
    def update_val(self, value, cl=None):
        self.ao.set_text(value)
        if cl:
            self.ao.set_c(cl)
        else:
            self.ao.set_c(self.cl)
            
    def show_secondary(self):
        if self.astr:  # if there is a 2ndary set of text
            self.astr.ao.set_c("green")

    def clear_secondary(self):
        if self.astr:  # if there is a 2ndary set of text
            self.astr.ao.set_c("white")

            
            
## For debug, put this in the notebook being debugged and be sure to set the out=out parameter
#out = widgets.Output(layout={'border': '1px solid black'})
#out            

class plt_network():
    
    def __init__(self, fn, image, out=None):
        self.out = out # debug
        #with self.out:
        #    print("hello world")
        img = plt.imread(image)
        self.fig, self.ax = plt.subplots(figsize=self.sizefig(img))
        boxes = fn(self.ax)
        self.boxes = boxes
        widgvis(self.fig)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.imshow(img)
        self.fig.text(0.1,0.9, "Click in boxes to fill in values.")
        self.glist = []  # place to stash global things 
        self.san = []    # selected annotation
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.axreveal = plt.axes([0.55, 0.02, 0.15, 0.075]) #[left, bottom, width, height]
        self.axhide   = plt.axes([0.76, 0.02, 0.15, 0.075])
        self.breveal  = Button(self.axreveal, 'Reveal All')
        self.breveal.on_clicked(self.reveal_values)
        self.bhide    = Button(self.axhide, 'Hide All')
        self.bhide.on_clicked(self.hide_values)
        #plt.show()

    def sizefig(self,img):
        iy,ix,iz = np.shape(img)
        if 10/5 < ix/iy:   # if x is the limiting size
            figx = 10
            figy = figx*iy/ix
        else:
            figy = 5
            figx = figy*ix/iy
        return(figx,figy)
       
    def updateval(self, event):
        #with self.out:  #debug
        #    print(event)
        box = self.san[0]
        num_format = re.compile(r"[+-]?\d+(?:\.\d+)?")
        isnumber = re.match(num_format,event)
        if not isnumber:
            box.update_val('?','red')
        else:
            #with self.out:
            #    print(event)
            newval = int(float(event)) if int(float(event)) == float(event) else float(event)
            newval = round(newval,2)
            #with self.out:
            #    print(newval, box.value, type(newval), type(box.value))
            if newval == box.value:
                box.show_secondary()
                box.update_val(round(newval,2))
            else:
                box.update_val(round(newval,2), 'red')
                box.clear_secondary()
        self.glist[0].remove()
        self.glist.clear()
        self.san.clear()

    # collects all clicks within diagram and dispatches
    def onclick(self, event):
        #with self.out:
        #    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #          ('double' if event.dblclick else 'single', event.button,
        #           event.x, event.y, event.xdata, event.ydata))
        if len(self.san) != 0: # already waiting for new value
            return
        inbox, box = inboxes((event.xdata, event.ydata), self.boxes)
        #with self.out:
        #    print(f" in box: {inbox, box}")
        if inbox:
            self.san.append(box) 
            #an.set_text(an.get_text() + "1") # debug
            graphBox = self.fig.add_axes([0.225, 0.02, 0.2, 0.075])  # [left, bottom, width, height]
            txtBox = TextBox(graphBox, "newvalue: ")
            txtBox.on_submit(self.updateval)
            self.glist.append(graphBox)
            self.glist.append(txtBox)
        return

    def reveal_values(self, event):
        for b in self.boxes:
            b.update_val(b.value)
            b.show_secondary()
        plt.draw()

    def hide_values(self, event):
        for b in self.boxes:
            b.update_val("?")
            b.clear_secondary()
        plt.draw()
        
#--------------------------------------------------------------------------


def config_nw0(ax):
    #"./images/C2_W2_BP_network0.PNG"

    w = 3
    a = 2+3*w
    J = a**2
    
    pass             ; dJ_dJ  = 1
    dJ_da = 2*a      ; dJ_da = dJ_dJ * dJ_da
    da_dw = 3        ; dJ_dw = dJ_da * da_dw

    box1 = abox(ax, round(a,2), 307,  140,  352, 100, (315, 128),'blue', None) # left, bottom, right, top, 
    box2 = abox(ax, round(J,2), 581,  138,  624, 100, (589, 128),'blue', None) 

    dJ_da_a = astring(ax, r"$\frac{\partial J}{\partial a}=$"+f"{dJ_da}", (291,186), "green")
    box3 = abox(ax, round(dJ_da,2), 545, 417, 588, 380, (553,407), 'green', dJ_da_a) 

    dJ_dw_a = astring(ax, r"$\frac{\partial J}{\partial w}=$"+f"{dJ_dw}", (60,186), "green")
    box4 = abox(ax, round(da_dw,2), 195, 421, 237, 380, (203,411), 'green', None)   
    box5 = abox(ax, round(dJ_dw,2), 265, 515, 310, 475, (273,505), 'green', dJ_dw_a)   

    boxes = [box1, box2, box3, box4, box5]
    
    return boxes   

def config_nw1(ax):
    # "./images/C2_W2_BP_Network1.PNG"

    x = 2
    w = -2
    b = 8
    y = 1
    
    c = w * x
    a = c + b
    d = a - y
    J = d**2/2
    
    pass             ; dJ_dJ = 1
    dJ_dd = 2*d/2    ; dJ_dd = dJ_dJ * dJ_dd
    dd_da = 1        ; dJ_da = dJ_dd * dd_da
    da_db = 1        ; dJ_db = dJ_da * da_db
    da_dc = 1        ; dJ_dc = dJ_da * da_dc
    dc_dw = x        ; dJ_dw = dJ_dc * dc_dw
    
    box1 = abox(ax, round(c,2), 330,  162,  382, 114, (338, 150),'blue', None) # left, bottom, right, top, 
    box2 = abox(ax, round(a,2), 636,  162,  688, 114, (644, 150),'blue', None) 
    box3 = abox(ax, round(d,2), 964,  162, 1015, 114, (972, 150),'blue', None) 
    box4 = abox(ax, round(J,2), 1266, 162, 1315, 114, (1274,150),'blue', None) 

    dJ_dd_a = astring(ax, r"$\frac{\partial J}{\partial d}=$"+f"{dJ_dd}", (967,208), "green")
    box5 = abox(ax, round(dJ_dd,2), 1222, 488, 1275, 441, (1230,478), 'green', dJ_dd_a) 

    dJ_da_a = astring(ax, r"$\frac{\partial J}{\partial a}=$"+f"{dJ_da}", (615,208), "green")
    box6 = abox(ax, round(dd_da,2), 900, 383, 951,  333, (908,373), 'green', None)   
    box7 = abox(ax, round(dJ_da,2), 988, 483, 1037, 441, (996,473), 'green', dJ_da_a)   

    dJ_dc_a = astring(ax, r"$\frac{\partial J}{\partial c}=$"+f"{dJ_dc}", (337,208), "green")
    box8 = abox(ax, round(da_dc,2),  570, 380, 620, 333, (578,370), 'green', None)   
    box9 = abox(ax, round(dJ_dc,2),  638, 467, 688, 419, (646,457), 'green', dJ_dc_a)   

    dJ_db_a = astring(ax, r"$\frac{\partial J}{\partial b}=$"+f"{dJ_dc}", (474,252), "green")
    box10 = abox(ax, round(da_db,2), 563, 582, 615, 533, (571,572), 'green', None)   
    box11 = abox(ax, round(dJ_db,2), 630, 677, 684, 630, (638,667), 'green', dJ_db_a)   

    dJ_dw_a = astring(ax, r"$\frac{\partial J}{\partial w}=$"+f"{dJ_dw}", (60,208), "green")
    box12 = abox(ax, round(dc_dw,2), 191, 379, 341, 332, (199,369), 'green', None)   
    box13 = abox(ax, round(dJ_dw,2), 266, 495, 319, 448, (274,485), 'green', dJ_dw_a)   

    boxes = [box1, box2, box3, box4, box5, box6, box7, box8, box9, box10, box11, box12, box13]

    return boxes   
            
#not used
def config_nw2():
    x0 = 1
    x1 = 2
    w0 = -2
    w1 = 3
    b  = -4
    y  = 1
    d  = x0 * w0
    e  = x1 * w1
    f  = d+e+b
    g  = -f
    h  = np.exp(g)
    i  = h+1
    a  = 1/i
    k  = y-a
    L  = k**2

    pass             ; dL_dL  = 1
    dL_dk = 2*k      ; dL_dk  = dL_dL * dL_dk
    dk_da = -1       ; dL_da  = dL_dk * dk_da
    da_di = -1/i**2  ; dL_di  = dL_da * da_di
    di_dh = 1        ; dL_dh  = dL_di * di_dh
    dh_dg = exp(g)   ; dL_dg  = dL_dh * dh_dg
    dg_df = -1       ; dL_df  = dL_dg * dg_df
    df_dd = 1        ; dL_dd  = dL_df * df_dd
    df_de = 2        ; dL_de  = dL_df * df_de
    df_db = 1        ; dL_db  = dL_df * df_db
    dd_dw0 = 1       ; dL_dw0 = dL_dd * dd_dw0
    de_dw1 = 2       ; dL_dw1 = dL_de * de_dw1

    an1 = avalue(round(d,2), (270,265), 'blue')
    an2 = avalue(round(e,2), (270,350), 'blue')
    an3 = avalue(round(f,2), (400,315), 'blue')
    an4 = avalue(round(g,2), (540,315), 'blue')
    an5 = avalue(round(h,2), (650,315), 'blue')
    an6 = avalue(round(i,2), (760,315), 'blue')
    an7 = avalue(round(a,2), (890,315), 'blue')
    an8 = avalue(round(k,2), (1015,315), 'blue')
    an9 = avalue(round(L,2), (1120,315), 'blue')
    bn1 = avalue(round(dL_dd,2), (260,300),  'green')   #d
    bn2 = avalue(round(dL_de,2), (270,385),  'green')   #e
    bn3 = avalue(round(dL_df,2), (408,350),  'green')   #f
    bn4 = avalue(round(dL_dg,2), (540,350),  'green')   #g
    bn5 = avalue(round(dL_dh,2), (650,350),  'green')   #h
    bn6 = avalue(round(dL_di,2), (760,350),  'green')   #i
    bn7 = avalue(round(dL_da,2), (890,350),  'green')   #a
    bn8 = avalue(round(dL_dk,2), (1015,350), 'green')   #k
    bn9 = avalue(round(dL_dw0,2), (210,300), 'green')   #w0
    bn10 = avalue(round(dL_dw1,2), (205,440),'green')   #w1
    bn11 = avalue(round(dL_db,2), (345,385), 'green')   #b

    anotes = [an1, an2, an3, an4, an5, an6, an7, an8, an9,
              bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8, bn9, bn10, bn11]

    box1 = abox(r"$\frac{\partial v}{\partial t}$", 943, 347, 980, 310, (980,300))
    boxes = [box1]

    fn = "./images/C2_W2_BP_bkground.PNG"
    return fn, anotes, boxes