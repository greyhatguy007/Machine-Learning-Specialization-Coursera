import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0', dldarkblue =  '#0D5BDC', dlmedblue='#4285F4')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; dldarkblue =  '#0D5BDC'; dlmedblue='#4285F4'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('./deeplearning.mplstyle')


def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

def plt_act_trio():
    X = np.linspace(-5,5,100)
    fig,ax = plt.subplots(1,3, figsize=(6,2))
    widgvis(fig)
    ax[0].plot(X,tf.keras.activations.linear(X))
    ax[0].axvline(0, lw=0.3, c="black")
    ax[0].axhline(0, lw=0.3, c="black")
    ax[0].set_title("Linear")
    ax[1].plot(X,tf.keras.activations.sigmoid(X))
    ax[1].axvline(0, lw=0.3, c="black")
    ax[1].axhline(0, lw=0.3, c="black")
    ax[1].set_title("Sigmoid")
    ax[2].plot(X,tf.keras.activations.relu(X))
    ax[2].axhline(0, lw=0.3, c="black")
    ax[2].axvline(0, lw=0.3, c="black")
    ax[2].set_title("ReLu")
    fig.suptitle("Common Activation Functions", fontsize=14)
    fig.tight_layout(pad=0.2)
    plt.show()

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def plt_ex1():
    X = np.linspace(0,2*np.pi, 100)
    y = np.cos(X)+1
    y[50:100]=0
    fig,ax = plt.subplots(1,1, figsize=(2,2))
    widgvis(fig)
    ax.set_title("Target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(X,y)
    fig.tight_layout(pad=0.1)
    plt.show()
    return(X,y)
 
def plt_ex2():
    X = np.linspace(0,2*np.pi, 100)
    y = np.cos(X)+1
    y[0:49]=0
    fig,ax = plt.subplots(1,1, figsize=(2,2))
    widgvis(fig)
    ax.set_title("Target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(X,y)
    fig.tight_layout(pad=0.1)
    plt.show()
    return(X,y)

def gen_data():
    X = np.linspace(0,2*np.pi, 100)
    y = np.cos(X)+1
    X=X.reshape(-1,1)
    return(X,y)

def plt_dual(X,y,yhat):
    fig,ax = plt.subplots(1,2, figsize=(4,2))
    widgvis(fig)
    ax[0].set_title("Target")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].plot(X,y)
    ax[1].set_title("Prediction")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].plot(X,y)
    ax[1].plot(X,yhat)
    fig.tight_layout(pad=0.1)
    plt.show()

def plt_act1(X,y,z,a):
    fig,ax = plt.subplots(1,3, figsize=(6,2.5))
    widgvis(fig)
    ax[0].plot(X,y,label="target")
    ax[0].axvline(0, lw=0.3, c="black")
    ax[0].axhline(0, lw=0.3, c="black")
    ax[0].set_title("y - target")
    ax[1].plot(X,y, label="target")
    ax[1].plot(X,z, c=dlc["dldarkred"],label="z")
    ax[1].axvline(0, lw=0.3, c="black")
    ax[1].axhline(0, lw=0.3, c="black")
    ax[1].set_title(r"$z = w \cdot x+b$")
    ax[1].legend(loc="upper center")
    ax[2].plot(X,y, label="target")
    ax[2].plot(X,a, c=dlc["dldarkred"],label="ReLu(z)")
    ax[2].axhline(0, lw=0.3, c="black")
    ax[2].axvline(0, lw=0.3, c="black")
    ax[2].set_title("max(0,z)")
    ax[2].legend()
    fig.suptitle("Role of Non-Linear Activation", fontsize=12)
    fig.tight_layout(pad=0.22)
    return(ax)


def plt_add_notation(ax):
    ax[1].annotate(text = "matches\n here", xy =(1.5,1.0), 
                   xytext = (0.1,-1.5), fontsize=9,
                  arrowprops=dict(facecolor=dlc["dlpurple"],width=2, headwidth=8))
    ax[1].annotate(text = "but not\n here", xy =(5,-2.5), 
                   xytext = (1,-3), fontsize=9,
                  arrowprops=dict(facecolor=dlc["dlpurple"],width=2, headwidth=8))
    ax[2].annotate(text = "ReLu\n 'off'", xy =(2.6,0), 
                   xytext = (0.1,0.1), fontsize=9,
                  arrowprops=dict(facecolor=dlc["dlpurple"],width=2, headwidth=8))

def compile_fit(model,X,y):
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    model.fit(
        X,y,
        epochs=100,
        verbose = 0
    )
    l1=model.get_layer("l1")
    l2=model.get_layer("l2")
    w1,b1 = l1.get_weights()
    w2,b2 = l2.get_weights()
    return(w1,b1,w2,b2)

def plt_model(X,y,yhat_pre, yhat_post):
    fig,ax = plt.subplots(1,3, figsize=(8,2))
    widgvis(fig)
    ax[0].set_title("Target")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].plot(X,y)
    ax[1].set_title("Prediction, pre-training")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].plot(X,y)
    ax[1].plot(X,yhat_pre)
    ax[2].set_title("Prediction, post-training")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].plot(X,y)
    ax[2].plot(X,yhat_post)
    fig.tight_layout(pad=0.1)
    plt.show()

def display_errors(model,X,y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    doo = yhat != y[:,0]
    idxs = np.where(yhat != y[:,0])[0]
    if len(idxs) == 0:
        print("no errors found")
    else:
        cnt = min(8, len(idxs))
        fig, ax = plt.subplots(1,cnt, figsize=(5,1.2))
        fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.80]) #[left, bottom, right, top]
        widgvis(fig)

        for i in range(cnt):
            j = idxs[i]
            X_reshaped = X[j].reshape((20,20)).T

            # Display the image
            ax[i].imshow(X_reshaped, cmap='gray')

            # Predict using the Neural Network
            prediction = model.predict(X[j].reshape(1,400))
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display the label above the image
            ax[i].set_title(f"{y[j,0]},{yhat}",fontsize=10)
            ax[i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
    return(len(idxs))

def display_digit(X):
    """ display a single digit. The input is one digit (400,). """
    fig, ax = plt.subplots(1,1, figsize=(0.5,0.5))
    widgvis(fig)
    X_reshaped = X.reshape((20,20)).T
    # Display the image
    ax.imshow(X_reshaped, cmap='gray')
    plt.show()
