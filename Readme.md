This code accompanies Mott (2022): "Life-Cycle Student Debt and Asset Prices."

---
# The Neural Network
Let the neural network be represented by the function $\Gamma(\cdot|\Theta)$, where $\Theta$ is the parameters of the model (weights and biases). The function maps $\Gamma: \mathbb{X} \to \mathbb{Y}$ where $\mathbb{X} \subseteq \mathbb{R}^{2\times J\times (L-1) + J\times L + 2}$ and $\mathbb{Y} \subseteq \mathbb{R}^{2\times J\times (L-1) + 2 }$. Denote elements of $\mathbb{X} \ni X$ and $\mathbb{Y} \ni Y$ so that $\Gamma(X|\Theta)=Y$.

$$\begin{aligned}
\Gamma&: & \mathbb{X} &\to \mathbb{Y} \\
\Gamma&: & \begin{pmatrix}
\left[\begin{array}{c}
[e_{i,j,t-1} ]_{i\leq L-1}\\
[b_{i,j,t-1} ]_{i\leq L-1}\\
[y_{i,j,t-1} ]_{i\leq L-1}
\end{array}\right]_{j}\\
Y_t \\
\delta_t
\end{pmatrix} &\mapsto 
\begin{pmatrix}
\left[\begin{array}{c}
[e_{i,j,t} ]_{i\leq L-1}\\
[b_{i,j,t} ]_{i\leq L-1}
\end{array}\right]_{j}\\
p_t \\
q_t
\end{pmatrix}
\end{aligned}$$

For calculations of equilibrium conditions, I sometimes wish to manipulate the input and output vectors to matrices according to the following transformations:

Reformatting input:

$$ \begin{pmatrix}
\left[\begin{array}{c}
[e_{i,j,t-1} ]_{i\leq L-1}\\
[b_{i,j,t-1} ]_{i\leq L-1}\\
[y_{i,j,t-1} ]_{i\leq L-1}
\end{array}\right]_{j}\\
Y_t \\
\delta_t
\end{pmatrix} \to \begin{pmatrix}
\left[\begin{array}{c}
[e_{i,j,t-1} ]_{i\leq L-1}\\
[b_{i,j,t-1} ]_{i\leq L-1}\\
[y_{i,j,t-1} ]_{i\leq L-1}\\
Y_t\\
\delta_t
\end{array}\right]_{j}^{\mathrm T} 
\end{pmatrix} =
\begin{pmatrix}
\begin{array}{c}
[e_{i,j=0,t-1} ]_{i\leq L-1} & \ldots & [e_{i,j=J-1,t-1} ]_{i\leq L-1}\\
[b_{i,j=0,t-1} ]_{i\leq L-1} & \ldots & [b_{i,j=J-1,t-1} ]_{i\leq L-1}\\
[y_{i,j=0,t-1} ]_{i\leq L-1} & \ldots & [y_{i,j=J-1,t-1} ]_{i\leq L-1}\\
Y_t       & \ldots & Y_t\\
\delta_t  & \ldots & \delta_t
\end{array}
\end{pmatrix} $$

Reformatting output:

$$ \begin{pmatrix}
\left[\begin{array}{c}
[e_{i,j,t} ]_{i\leq L-1}\\
[b_{i,j,t} ]_{i\leq L-1}
\end{array}\right]_{j}\\
p_t \\
q_t
\end{pmatrix} \to \begin{pmatrix}
\left[\begin{array}{c}
[e_{i,j,t} ]_{i\leq L-1}\\
[b_{i,j,t} ]_{i\leq L-1}\\
p_t\\
q_t
\end{array}\right]_{j}^{\mathrm T}
\end{pmatrix} =
\begin{pmatrix}
\begin{array}{c}
[e_{i,j=0,t} ]_{i\leq L-1} & \ldots & [e_{i,j=J-1,t} ]_{i\leq L-1}\\
[b_{i,j=0,t} ]_{i\leq L-1} & \ldots & [b_{i,j=J-1,t} ]_{i\leq L-1}\\
p_t & \ldots & p_t\\
q_t & \ldots & q_t
\end{array}
\end{pmatrix} $$

---
# Training Loop
## Forward Simulation
At stage $n$ of the training, parameter values are $\Theta_n$. Using the neural network $\Gamma(X|\Theta_n)$, compute the temporary equilibrium time series for $T$ periods. 

## Back Propagation
Now given the training data $\{X_t\}_t$ and $\{Y_t\}_t$ for $t \in \{0,\ldots,T-1\}$, update the parameters according to the loss function. 

$$\mathcal{L}(X,Y|\Theta_n) = $$

$$\frac{\partial \mathcal{L}}{\partial \Theta_n }$$
