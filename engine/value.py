class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0 # We will use this later for calculus/gradients
        self._prev = set(_children) # The pointers to the children/parent nodes
        self._op = _op # The math operation (+, *)

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # Wrap the other number in a Value object if it isn't one already
        other = other if isinstance(other, Value) else Value(other)
        # Create a new node, passing in the current node and the 'other' node as children
        out = Value(self.data + other.data, (self, other), '+')
        return out