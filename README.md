# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model

<img width="754" height="458" alt="image" src="https://github.com/user-attachments/assets/3ff83b4e-a19e-4fd6-a48f-1b097ba52a05" />

## DESIGN STEPS

### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Sivasakthi S

### Register Number: 212225040418

```
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here

python

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(71)
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Dataset')
plt.show()

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 50
losses = []

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
          f"Weight={model.linear.weight.item():.4f}, "
          f"Bias={model.linear.bias.item():.4f}")

plt.plot(range(1, epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss vs Epoch')
plt.show()

x_line = np.linspace(0, 50, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
y_line = current_weight * x_line + current_bias

plt.scatter(X, y, label='Original Data')
plt.plot(x_line, y_line, 'r', label='Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

new_input = torch.tensor([[60.0]])
prediction = model(new_input)

print("\nSample Input:", new_input.item())
print("Predicted Output:", prediction.item())



# Initialize the Model, Loss Function, and Optimizer

torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Initialize the Model, Loss Function, and Optimizer

```

### Dataset Information

<img width="568" height="450" alt="image" src="https://github.com/user-attachments/assets/57971fb1-c687-43e1-a2b4-b1723ed71ae3" />

### OUTPUT
## Training Loss Vs Iteration Plot

<img width="578" height="454" alt="image" src="https://github.com/user-attachments/assets/d4b23eec-1e94-4712-bc16-15dec4dffee2" />

## Best Fit line plot

<img width="568" height="446" alt="image" src="https://github.com/user-attachments/assets/eb900d4e-b806-4756-89fb-629cecbf58db" />

### New Sample Data Prediction

<img width="692" height="226" alt="image" src="https://github.com/user-attachments/assets/422a1e45-b87a-4068-9800-aaabc74a4032" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
