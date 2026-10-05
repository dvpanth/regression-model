import numpy as np
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self,alpha=0.01,max_iter=10000,tolerance=1e-9):
        self.w = None
        self.b = None
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
    def predict(self,X):
        return (X@self.w)+self.b
    def calculate_error(self,y_pred,y_actual):
        return np.array(y_actual - y_pred)
    def calculate_loss(self,err,n):
        return (1/n)*(err.T@err) # This is the same as 1\n * sum((err_i)^2)
    def step(self,n,err,X):
        # Step down
        dw = (-2/n)*X.T@(err)
        db = (-2/n)*np.sum(err)
        # Update parameters
        self.w = self.w - (self.alpha*dw)
        self.b = self.b - (self.alpha*db)
    def fit(self, X_train, y_train):
        n,p=X_train.shape
        self.w=np.zeros(p)
        self.b=0.
        y_pred = self.predict(X=X_train)
        err = self.calculate_error(y_pred=y_pred,y_actual=y_train)
        loss = self.calculate_loss(err=err,n=n)
        for i in range(self.max_iter):
            previous_loss = loss
            self.step(n=n, err=err,X=X_train)
            y_pred=self.predict(X=X_train)
            err=self.calculate_error(y_pred=y_pred,y_actual=y_train)
            loss = self.calculate_loss(err=err,n=n)
            if abs(loss-previous_loss)<self.tolerance:
                break

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
    n,p=X_train.shape
    model.fit(X_train=X_train, y_train=y_train)
    prediction = model.predict(X_train)
    err = model.calculate_error(y_pred=prediction, y_actual=y_train)
    loss = model.calculate_loss(err=err, n=n)
    print(f"Loss after training: {loss}\nWeight: {model.w}\nBias: {model.b}")
    plt.scatter(y_train, prediction)
    low = min(y_train.min(), prediction.min())
    high = max(y_train.max(),prediction.max())
    plt.plot([low, high], [low, high])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

if __name__ == '__main__':
    main()
