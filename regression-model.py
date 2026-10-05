import numpy as np
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self, w = np.array([0.,0.,0.]), b = 0, alpha=0.01):
        self.w = w
        self.b = b
        self.alpha = alpha
    def predict(self, X):
        return (X@self.w)+self.b
    def get_size(self, y):
        return len(y)
    def calculate_error(self, y_pred, y_actual):
        return np.array(y_actual - y_pred)
    def calculate_loss(self,err, n):
        return (1/n)*(err.T@err) # This is the same as 1\n * sum((err_i)^2)
    def step(self,n,err,X):
        # Calculate gradients
        dw = (-2/n)*X.T@(err)
        db = (-2/n)*sum(err)
        # Update parameters
        self.w = self.w - (self.alpha*dw)
        self.b = self.b - (self.alpha*db)

def create_data():
    np.random.seed(42)
    n = 100
    p = 3
    X_train = np.random.randn(n,p)
    true_w = np.array([2., -3., 1.5])
    true_b = 5.
    noise = np.random.normal(0, 1, n)
    y_train = X_train @ true_w + true_b + noise
    return X_train, y_train

def main():
    X_train, y_train = create_data()
    model = RegressionModel()
    y_pred = model.predict(X=X_train)
    print(y_pred)
    err = model.calculate_error(y_pred=y_pred, y_actual=y_train)
    n = model.get_size(y=y_train)
    loss = model.calculate_loss(n=n, err=err)
    loss_i = 0
    loss_f = loss
    max_i = 10000
    tolerance = 1e-9
    i=0
    while abs(loss_f - loss_i)>tolerance and i<max_i:
        model.step(n=n, err=err, X=X_train)
        y_pred = model.predict(X=X_train)
        loss_i = loss_f
        err = model.calculate_error(y_pred=y_pred, y_actual=y_train)
        loss_f = model.calculate_loss(err, n)
        i+=1
    print(f"Loss after {i} rounds: {loss_f}\nWeight: {model.w}\nBias: {model.b}")
    plt.scatter(y_train, y_pred)
    low = min(y_train.min(), y_pred.min())
    high = max(y_train.max(),y_pred.max())
    plt.plot([low, high], [low, high])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

if __name__ == '__main__':
    main()
