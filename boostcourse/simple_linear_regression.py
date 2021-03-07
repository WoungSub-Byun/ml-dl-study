import tensorflow as tf

tf.enable_eager_execution()

# Data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)

learning_rate = 0.01

for i in range(101):  # W, b update
    # Gradient descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # 평균값 구하기
    W_grad, b_grad = tape.gradient(cost, [W, b])  # 미분값(기울기) 구하기
    W.assign_sub(learning_rate * W_grad)  # tensor 값 업데이트
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# Epoch: for문 안의 코드가 한 번 실행되는 횟수
# A.assign_sub(B)
# - A값 업데이트
# - A = A - B와 같은 역할
# - 위쪽의 식을 그대로 쓰면 실제 tensor에는 영향이 가지 않기 때문에 assign_sub 메서드를 이용한다.