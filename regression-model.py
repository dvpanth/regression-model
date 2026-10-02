import numpy as np
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self, w = 0, b = 0, alpha=0.01):
        self.w = w
        self.b = b
        self.alpha = alpha
    def predict(self, X_train):
        return [self.w*val+self.b for val in X_train]
    def calculate_loss(self,y_pred,y_actual):
        return (1/len(y_actual))*sum((y_actual - y_pred)**2)
    def train(self,X_train,y_actual,y_pred):
        # Calculate gradients
        dw = (-2/len(y_actual))*sum(X_train*(y_actual-y_pred))
        db = (-2/len(y_actual))*sum(y_actual-y_pred)
        # Update parameters
        self.w = self.w - (self.alpha*dw)
        self.b = self.b - (self.alpha*db)

def main():
    X_train = np.array([1,2,3,4,5,6,7,8,9,10])
    y_train = np.array([3,5,8,11,12,13,16,17,19,22])
    plt.scatter(x=X_train, y=y_train)
    model = RegressionModel(alpha=0.01)
    y_pred = model.predict(X_train=X_train)
    tolerance = 1e-9
    max_i = 100000
    loss_i=0
    loss_f=model.calculate_loss(y_pred=y_pred, y_actual=y_train)
    i=0

    while abs(loss_f - loss_i)>tolerance and i<max_i:
        model.train(X_train=X_train, y_actual=y_train, y_pred=y_pred)
        y_pred = model.predict(X_train=X_train)
        loss_i = loss_f
        loss_f = model.calculate_loss(y_pred=y_pred, y_actual=y_train)
        i+=1

    print(f"Loss after {i} rounds: {loss_f}\nWeight: {model.w}\nBias: {model.b}")
    plt.plot(X_train, y_pred, color='red', linewidth=2, label='Arbitrary Model')
    plt.show()
if __name__ == '__main__':
    main()
