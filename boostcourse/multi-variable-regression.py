import tensorflow as tf
import numpy as np


def multi_variable_regression_base():
    x1 = [73.0, 93.0, 89.0, 96.0, 73.0]
    x2 = [80.0, 88.0, 89.0, 98.0, 66.0]
    x3 = [75.0, 93.0, 90.0, 100.0, 70.0]
    Y = [152.0, 185.0, 180.0, 196.0, 142.0]

    w1 = tf.Variable(tf.random_normal([1]))
    w2 = tf.Variable(tf.random_normal([1]))
    w3 = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    learning_rate = 0.00001

    for i in range(1000 + 1):
        with tf.GradientTape() as tape:
            hypothesis = w1 * x2 + w2 * x2 + w3 * x3 + b
            cost = tf.reduce_mean(tf.square(hypothesis - Y))
        w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

        w1.assign_sub(learning_rate * w1_grad)
        w2.assign_sub(learning_rate * w2_grad)
        w3.assign_sub(learning_rate * w3_grad)
        b.assign_sub(learning_rate * b_grad)

        if i % 50 == 0:
            print("{:5} | {:12.4f}".format(i, cost.numpy()))


# 위의 multi_variable_regression_base함수와 multi_variable_regression_matrix함수는 결과적으로 같은 함수입니다.
# 하지만 아래의 함수는 행렬(matrix)를 사용한 점곱연산(Dot production)을 하였기 때문에 훨씬 더 간결하고 가독성이 좋습니다.
def multi_variable_regression_matrix():
    data = np.array(
        [
            [73.0, 80.0, 75.0, 152.0],
            [93.0, 88.0, 93.0, 185.0],
            [89.0, 88.0, 93.0, 180.0],
            [96.0, 98.0, 100.0, 196.0],
            [73.0, 66.0, 70.0, 142.0],
        ],
        dtype=np.float32,
    )

    x = data[:, :-1]
    y = data[:, [-1]]

    W = tf.Variable(tf.random_normal([3, 1]))
    b = tf.Variable(tf.random_normal([1]))

    def predict(X):
        return tf.matmul(X, W) + b  # 점곱 연산

    n_epochs = 2000
    for i in range(n_epochs + 1):
        with tf.GradientTape() as tape:
            cost = tf.reduce_mean(tf.square(predict(X) - Y))

            W_grad, b_grad = tape.gradient(cost, [W, b])

            W.assign_sub(learning_rate * W_grad)
            b.assign_sub(learning_rate * b_grad)

            if i % 100 == 0:
                print("{:5} | {:10.4f}".format(i, cost.numpy()))
