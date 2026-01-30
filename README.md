# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2.Write a function compute Cost to generate the cost function.
3.Perform iterations of gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S AVAN ARUL
RegisterNumber: 212225040036
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x=data["R&D Spend"].values
y=data["Profit"].values
x_mean=np.mean(x)
x_std=np.std(x)
x=(x-x_mean)/x_std
w = 0.0          
b = 0.0          
alpha = 0.01     
epochs = 100
n = len(x)
losses=[]
for i in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w = w - alpha * dw
    b = b - alpha * db
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line",color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")


plt.tight_layout()
plt.show()



```

## Output:
<img width="747" height="297" alt="Screenshot 2026-01-30 144729" src="https://github.com/user-attachments/assets/42b0354c-5e70-447f-b153-cdd5cdbd841f" />
<img width="411" height="47" alt="Screenshot 2026-01-30 144803" src="https://github.com/user-attachments/assets/f707f255-d373-4874-94bc-56635173da9e" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

