# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import gzip
from collections import OrderedDict

# ====================== 核心操作类 ======================
class Layer:
    def __init__(self):
        self.params = OrderedDict()
        self.grads = OrderedDict()
        self.optimizable = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

class Linear(Layer):
    def __init__(self, in_dim, out_dim, weight_decay=False, weight_decay_lambda=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        # He初始化
        scale = np.sqrt(2. / in_dim)
        self.params['W'] = np.random.randn(in_dim, out_dim) * scale
        self.params['b'] = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @ self.params['W'] + self.params['b']

    def backward(self, grad):
        self.grads['W'] = self.x.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']
        
        return grad @ self.params['W'].T

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False

    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        return grad  # 通常与CrossEntropyLoss合并计算

class MultiCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, y_pred, y_true):
        m = y_true.shape[0]
        grad = y_pred.copy()
        grad[range(m), y_true] -= 1
        return grad / m

# ====================== 模型定义 ======================
class Model_MLP(Layer):
    def __init__(self, size_list=[784, 256, 64, 10], act_func='ReLU', lambda_list=[0.01, 0.01, 0.01]):
        super().__init__()
        self.size_list = size_list
        self.layers = []
        self.training = True
        
        for i in range(len(size_list) - 1):
            layer = Linear(size_list[i], size_list[i+1], 
                         weight_decay=True, 
                         weight_decay_lambda=lambda_list[i] if i < len(lambda_list) else 0)
            self.layers.append(layer)
            
            if i < len(size_list) - 2:
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
            else:
                self.layers.append(Softmax())

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

# ====================== 优化器和调度器 ======================
class SGD:
    def __init__(self, lr=0.01, model=None):
        self.lr = lr
        self.model = model

    def step(self):
        for layer in self.model.layers:
            if not layer.optimizable:
                continue
            for param in layer.params:
                layer.params[param] -= self.lr * layer.grads[param]

class MomentGD:
    def __init__(self, lr=0.01, model=None, mu=0.9):
        self.lr = lr
        self.model = model
        self.mu = mu
        self.velocity = OrderedDict()
        
        # 初始化速度项
        for layer in self.model.layers:
            if layer.optimizable:
                for param in layer.params:
                    self.velocity[id(layer), param] = np.zeros_like(layer.params[param])

    def step(self):
        for layer in self.model.layers:
            if not layer.optimizable:
                continue
            for param in layer.params:
                key = (id(layer), param)
                self.velocity[key] = self.mu * self.velocity[key] - self.lr * layer.grads[param]
                layer.params[param] += self.velocity[key]

class StepLR:
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

# ====================== 训练工具 ======================
class RunnerM:
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.metric = metric
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.train_loss = []
        self.dev_loss = []
        self.train_acc = []
        self.dev_acc = []

    def train(self, train_data, dev_data, num_epochs=10, log_iters=100):
        train_X, train_y = train_data
        dev_X, dev_y = dev_data
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            indices = np.random.permutation(len(train_X))
            for i in range(0, len(train_X), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = train_X[batch_idx]
                y_batch = train_y[batch_idx]
                
                # 前向传播
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                
                # 反向传播
                grad = self.loss_fn.backward(y_pred, y_batch)
                self.model.backward(grad)
                self.optimizer.step()
            
            # 评估
            train_loss, train_acc = self.evaluate((train_X, train_y))
            dev_loss, dev_acc = self.evaluate((dev_X, dev_y))
            
            self.train_loss.append(train_loss)
            self.dev_loss.append(dev_loss)
            self.train_acc.append(train_acc)
            self.dev_acc.append(dev_acc)
            
            if self.scheduler:
                self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    def evaluate(self, data):
        X, y = data
        self.model.eval()
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        acc = np.mean(np.argmax(y_pred, axis=1) == y)
        return loss, acc

# ====================== 主程序 ======================
def load_mnist():
    # 加载训练数据
    with gzip.open('./dataset/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        train_X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    
    with gzip.open('./dataset/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        train_y = np.frombuffer(f.read(), dtype=np.uint8)

    # 加载测试数据
    with gzip.open('./dataset/MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        test_X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    
    with gzip.open('./dataset/MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        test_y = np.frombuffer(f.read(), dtype=np.uint8)

    # 归一化并划分验证集
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    
    # 标准化
    mean, std = train_X.mean(), train_X.std()
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std
    
    # 划分验证集
    val_size = int(0.1 * len(train_X))
    dev_X, dev_y = train_X[:val_size], train_y[:val_size]
    train_X, train_y = train_X[val_size:], train_y[val_size:]
    
    return (train_X, train_y), (dev_X, dev_y), (test_X, test_y)

def plot_results(runner):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(runner.train_loss, label='Train')
    plt.plot(runner.dev_loss, label='Dev')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(runner.train_acc, label='Train')
    plt.plot(runner.dev_acc, label='Dev')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

def main():
    np.random.seed(42)
    
    # 加载数据
    (train_X, train_y), (dev_X, dev_y), (test_X, test_y) = load_mnist()
    
    # 创建模型
    model = Model_MLP(size_list=[784, 256, 64, 10], lambda_list=[0.01, 0.01, 0.01])
    
    # 优化器配置
    optimizer = MomentGD(lr=0.01, model=model, mu=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 训练
    runner = RunnerM(model, optimizer, accuracy, MultiCrossEntropyLoss(), batch_size=64, scheduler=scheduler)
    runner.train((train_X, train_y), (dev_X, dev_y), num_epochs=20)
    
    # 测试
    test_loss, test_acc = runner.evaluate((test_X, test_y))
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # 绘制曲线
    plot_results(runner)

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

if __name__ == '__main__':
    main()