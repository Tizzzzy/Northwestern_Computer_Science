# HW 6 - Neural Nets


## The Objective

This assignment allows you to explore the role and importance of hyperparameters in the design and operation of neural networks (aka neural nets). The supplied code is for a Feedforward Neural Network (FFNN) that classifies handwritten digits in the famous MNIST dataset. Remember the 28 x 28 example digit we saw when learning about neural nets? We saw how it was flattened into a long string of pixel values before being fed to a FFNN for classification. The supplied code does the same. All you have to do is tweak the hyperparameters to see how that impacts classification and report your observations. 


## The Task

There's some minor setup involved (see **The Setup** below), after which here's what you need to do:

Identify the block of code with `# Hyperparameters` as a header. Notice that it lists the following hyperparameters and specifies some default settings for them:

* `Learning Rate` = 0.001
* `Batch Size` = 32
* `Epochs` = 10
* `Dropout Rate` = 0.2 (We haven't discussed this in class, but you can read up on it quickly on the Web.)
* `Hidden Layer Size` = 64

The first thing you should do is to run the code and observe the dynamic output in the output console of your IDE or your terminal ("dynamic" referring to the epochs shown one by one). Note that the output shows certain metrics: Training Loss (`loss`), Training Accuracy (`accuracy`), Validation Loss (`val_loss`), and Validation Accuracy (`val_accuracy`). Notice how they change over the epochs. Also shown at the very end is the Test Accuracy, preceded by a line showing the Test Loss (also called simply `loss`) and the Test Accuracy (also called simply `accuracy`).

Now, get tweaking! Start by tweaking one hyperparameter at a time and observe its effect on the training and test metrics. Use your intuition to guide how you change the hyperparameters. For example, changing the learning rate from 0.001 to 0.002 will likely not have a noticeable effect, but changing it to 0.01 or 0.1 certainly will. Once you get a good understanding of the individual hyperparameters, start experimenting by changing more than one of them at a time. For example, what happens if you lower the learning rate down to 0.0001 (that's 1/10th the default provided) but use 10 times as many nodes in the hidden layer as the default? What happens if you add 10 epochs of training or don't train for enough epochs? Does batch size make things interesting? How so? Let your curiosity drive and questions as you experiment. For such experiments, you should always keep a log of everything you do so you can do comparative analyses. That's a part and parcel of the experimentation step of the scientific method -- keep a detailed log of your experiments so you can report on them. For this assignment, however, you don't have to report on your entire suite of experiments. Read on to know how much to report on.


## The Report

Aside from any experimentation you do on your own, your task is to experiment with the hyperparameter settings shown in the two tables below **and keep a log**, for which you can simply add two rows to the tables below -- one row for **`Test Accuracy`** and one for **`Training Time`** in seconds. Based on those experiments, you will need to do two things:

1. Answer the 9 questions in the **Canvas** assignment titled "**`HW 6 - Neural Nets Responses`**," which is an un-timed quiz. Your log will prove helpful for this. This task will carry 1 point for each answer so **9 points total**.
2. Upload your log---as a table in a PDF---to the assignment on **Canvas** titled "**`HW 6 - Neural Nets`**." This task will carry **1 point**.

This assignment DOES NOT require pushing anything to GitHub. Both the aforementioned tasks must be completed on Canvas.

The tables below list the experiments you have to do, some with single hyperparameters changes, some with double or triple, as shown in color. For example, `Experiment 1` has two sub-experiments, the hyperparameters for each listed in each of the two columns under the experiment. Likewise, `Experiment 4` has three sub-experiments, the hyperparameters for each listed in each of the three columns under the experiment.

### Table 1: Experiments with Single Hyperparameter Changes
<img width="1013" alt="image" src="https://github.com/NUCS348/assignment-6-neural-nets/assets/13566261/fb1dca02-27f7-4547-ad82-d3da5a22b24c">

### Table 2: Experiments with Double and Triple Hyperparameter Changes
<img width="814" alt="image" src="https://github.com/NUCS348/assignment-6-neural-nets/assets/13566261/cee69215-e4c3-48e6-b5d0-fbd9932d70bc">


## The Setup

The supplied code was tested on a Mac running macOS Ventura 13.1 and Python 3.9 running in a virtual environment, but given its simplicity, it should run on any computer with a relatively recent operating system and a recent Python version. A couple of things that needed to be done to test the code:

1. TensorFlow was installed within the virtual environment using `pip install tensorflow`.
   
2. Within the folder where Python is installed, the `Install Certificates.command` file needed to be run to install the certificates provided with Python. Not executing this step was preventing the MNIST dataset from being downloaded because Python could not verify the SSL certificate of the server it was trying to connect to.


## The End

That's it! That's all you have to do. Completing this assignment should provide you with an understanding and an appreciation of neural net hyperparameters.
