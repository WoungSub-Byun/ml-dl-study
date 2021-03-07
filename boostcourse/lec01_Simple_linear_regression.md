# Lec02: Simple linear regression

> Francis Garton

_Regression toward the mean_
회귀는

### Linear regression

데이터를 가장 잘 대변하는 직선의 방정식을 찾는 것
y = ax + b

a = 직선의 기울기
b = 방정식의 y절편

즉, 선형회귀는 위의 직선의 방정식의 a, b값을 찾는 것

H(x) = Wx + b

-   H: Hypothesis
-   W: Weight
-   b: bias

Hypothesis: 가설

비용(cost): 찾은 직선의 방정식에서 실제 데이터의 y값과 비교한것

비용이 작을 수록 좋은 직선의 방정식이라고 할 수 있음.

그래프에서 그냥 비용을 보면 음수 값이 나올 수 있으므로

(H(x) - y) ^ 2

-   H(x): 예측값
-   y: 실제값

제곱하여 사용함

따라서, 선형회귀의 목적은 **Cost Function을 최소화 시키는 W, b값을 찾는 것** **_== 학습_**

### Gradient Descent

-   gradient : 경사
-   descent: : 하강법

cost function의 값을 최소화하는 알고리즘 중 하나
