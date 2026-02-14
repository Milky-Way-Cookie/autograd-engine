from nn import MLP

n = MLP(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] 

# The Training Loop
for k in range(20):
    
    # 1. Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # 2. Backward pass (Zero out old gradients first!)
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    # 3. Update the weights (Gradient Descent)
    learning_rate = 0.05
    for p in n.parameters():
        p.data += -learning_rate * p.grad
        
    print(f"Epoch {k} | Loss: {loss.data:.4f}")

print("\nFinal Predictions:")
print([y.data for y in ypred])