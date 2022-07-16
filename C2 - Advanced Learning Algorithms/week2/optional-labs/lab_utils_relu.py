import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('./deeplearning.mplstyle')
from matplotlib.widgets import Slider
from lab_utils_common import dlc

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    
    
def plt_base(ax):
    X = np.linspace(0, 3, 3*100)
    y = np.r_[ -2*X[0:100]+2, 1*X[100:200]-3+2, 3*X[200:300]-7+2 ]
    w00 = -2
    b00 =  2
    w01 =  0  #  1
    b01 =  0  # -1
    w02 =  0  #  2
    b02 =  0  # -4
    ax[0].plot(X, y, color = dlc["dlblue"], label="target")
    arts = []
    arts.extend( plt_yhat(ax[0], X, w00, b00, w01, b01, w02, b02) )
    _ = plt_unit(ax[1], X, w00, b00)   #Fixed
    arts.extend( plt_unit(ax[2], X, w01, b01) )
    arts.extend( plt_unit(ax[3], X, w02, b02) )
    return(X, arts)

def plt_yhat(ax, X, w00, b00, w01, b01, w02, b02):
    yhat = np.maximum(0, np.dot(w00, X) + b00) + \
            np.maximum(0, np.dot(w01, X) + b01) + \
            np.maximum(0, np.dot(w02, X) + b02)
    lp = ax.plot(X, yhat, lw=2, color = dlc["dlorange"], label="a2")
    return(lp)

def plt_unit(ax, X, w, b):
    z = np.dot(w,X) + b
    yhat = np.maximum(0,z)
    lpa = ax.plot(X, z,    dlc["dlblue"], label="z")
    lpb = ax.plot(X, yhat, dlc["dlmagenta"], lw=1, label="a")
    return([lpa[0], lpb[0]])

# if output is need for debug, put this in a cell and call ahead of time. Output will be below that cell.
#from ipywidgets import Output   #this line stays here
#output = Output()               #this line stays here
#display(output)                 #this line goes in notebook

def plt_relu_ex():
    artists = []

    fig = plt.figure()
    fig.suptitle("Explore Non-Linear Activation")

    gs = GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[2,1])
    ax = [ax1,ax2,ax3,ax4]
    
    widgvis(fig)
    #plt.subplots_adjust(bottom=0.35)

    axb2 = fig.add_axes([0.15, 0.10, 0.30, 0.03]) # [left, bottom, width, height]
    axw2 = fig.add_axes([0.15, 0.15, 0.30, 0.03])
    axb1 = fig.add_axes([0.15, 0.20, 0.30, 0.03])
    axw1 = fig.add_axes([0.15, 0.25, 0.30, 0.03])

    sw1 = Slider(axw1, 'w1', -4.0, 4.0, valinit=0, valstep=0.1)
    sb1 = Slider(axb1, 'b1', -4.0, 4.0, valinit=0, valstep=0.1)
    sw2 = Slider(axw2, 'w2', -4.0, 4.0, valinit=0, valstep=0.1)
    sb2 = Slider(axb2, 'b2', -4.0, 4.0, valinit=0, valstep=0.1)
    
    X,lp = plt_base(ax)
    artists.extend( lp )
    
    #@output.capture()
    def update(val):
        #print("-----------")
        #print(f"len artists {len(artists)}", artists)
        for i in range(len(artists)):
            artist = artists[i]
            #print("artist:", artist)
            artist.remove()
        artists.clear()
        #print(artists)
        w00 = -2
        b00 =  2
        w01 =  sw1.val  #  1
        b01 =  sb1.val  # -1
        w02 =  sw2.val  #  2
        b02 =  sb2.val  # -4
        artists.extend(plt_yhat(ax[0], X, w00, b00, w01, b01, w02, b02))
        artists.extend(plt_unit(ax[2], X, w01, b01) )
        artists.extend(plt_unit(ax[3], X, w02, b02) )
        #fig.canvas.draw_idle()
        
    sw1.on_changed(update)
    sb1.on_changed(update)
    sw2.on_changed(update)
    sb2.on_changed(update)

    ax[0].set_title(" Match Target ")
    ax[0].legend()
    ax[0].set_xlabel("x")
    ax[1].set_title("Unit 0 (fixed) ")
    ax[1].legend()
    ax[2].set_title("Unit 1")
    ax[2].legend() 
    ax[3].set_title("Unit 2")
    ax[3].legend()
    plt.tight_layout()

    plt.show()
    return([sw1,sw2,sb1,sb2,artists]) # returned to keep a live reference to sliders

