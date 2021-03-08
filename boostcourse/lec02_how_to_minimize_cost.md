# How to minimize cost?

> 복습

> -   Hypothesis
>     -   H(x) = Wx + b
> -   Cost
>     -   tf.reduce_mean(tf.square(hypothesis - y))

### Gradient Descent: 경사 하강법

-   how it works?
-   최초의 W,b 값을 지정
-   W,b 값이 최적이라고 판단될 때까지 W,b값을 변경한다.
-   Gradient: 미분을 통해 구해짐
