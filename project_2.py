import numpy as np
import matplotlib.pyplot as plt

class NonLinearRegression():
    def __init__(self, train_size, test_size, feature_para_1, feature_para_2, target_para_1, target_para_2):
        self.train_size = train_size
        self.test_size = test_size
        self.feature_para_1 = feature_para_1
        self.feature_para_2 = feature_para_2
        self.target_para_1 = target_para_1
        self.target_para_2 = target_para_2
        self.generate_data()
        self.preprocessing()
        self.non_linear_regression()
        self.plot_fitted_curves()

    def generate_data(self):
        np.random.seed(42)
        self.X_train = np.random.uniform(self.feature_para_1, self.feature_para_2, self.train_size)
        self.X_test = np.random.uniform(self.feature_para_1, self.feature_para_2, self.test_size)
        self.t_train = np.sin(2 * np.pi * self.X_train) + np.random.normal(self.target_para_1, self.target_para_2, self.train_size)
        self.t_test = np.sin(2 * np.pi * self.X_test) + np.random.normal(self.target_para_1, self.target_para_2, self.test_size)

    def preprocessing(self):
        self.x_train = self.X_train.reshape(-1, 1)
        self.x_test = self.X_test.reshape(-1, 1)
        self.t_train = self.t_train.reshape(-1, 1)
        self.t_test = self.t_test.reshape(-1, 1)

    def polynomial_features(self, X, M):
        return np.hstack([X ** i for i in range(M + 1)])

    def non_linear_regression(self):
        self.train_errors = []
        self.test_errors = []
        self.models = {}

        for M in range(0, 10):
            Phi_train = self.polynomial_features(self.x_train, M)
            Phi_test = self.polynomial_features(self.x_test, M)

            w = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ self.t_train
            self.models[M] = w

            y_train_pred = Phi_train @ w
            y_test_pred = Phi_test @ w

            train_error = np.mean((self.t_train - y_train_pred) ** 2)
            test_error = np.mean((self.t_test - y_test_pred) ** 2)

            self.train_errors.append(train_error)
            self.test_errors.append(test_error)

        self.plot_errors()

    def plot_errors(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(0, 10), self.train_errors, marker='o', label="Training Error")
        plt.plot(range(0, 10), self.test_errors, marker='s', label="Test Error")
        plt.xlabel("Polynomial Degree (M)")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.title(f"Training vs Test Error (N={self.train_size})")
        plt.show()

    def plot_fitted_curves(self):
        X_smooth = np.linspace(0, 1, 100).reshape(-1, 1)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for M in range(0, 10):
            Phi_smooth = self.polynomial_features(X_smooth, M)
            y_smooth = Phi_smooth @ self.models[M]

            ax = axes[M]
            ax.scatter(self.X_train, self.t_train, color='red', marker='o', alpha=0.6)
            ax.scatter(self.X_test, self.t_test, color='green', marker='x', alpha=0.6)
            ax.plot(X_smooth, y_smooth, 'b-')

            ax.text(0.05, 0.8, f"Train Error: {self.train_errors[M]:.3f}\nTest Error: {self.test_errors[M]:.3f}", 
                    transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            ax.set_xlabel("X")
            ax.set_ylabel("t")
            ax.set_title(f"Polynomial Fit (M={M})")

        
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label="Training Data"),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=8, label="Test Data"),
            plt.Line2D([0], [0], color='b', linewidth=2, label="Polynomial Fit")
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=10)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    NonLinearRegression(10, 100, 0, 1, 0, 0.3)
    NonLinearRegression(100, 100, 0, 1, 0, 0.3)
