from value import Value

# Create two leaf nodes
a = Value(2.0)
b = Value(3.0)

# Add them together to create a new parent node
c = a + b

print(f"Node C data: {c.data}")
print(f"Node C was created by: '{c._op}'")
print(f"Node C's children are: {c._prev}")