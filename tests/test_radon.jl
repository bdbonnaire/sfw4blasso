import Pkg
Pkg.activate("..")
push!(LOAD_PATH,"../src")

using Plots 
using RadonKA
using blasso
using sfw

# %% Creating the Line

# Model constants
sigma=[1., 1.];
M = 32.;
px=range(-M, M);
px=collect(px);
py=px;
angle_max = π/3;
angle_min = - angle_max;
# Option solver
options=sfw.sfw_options();
# Load kernel attributes
kernel=blasso.setGaussLinesKernel(px,py,sigma,angle_min,angle_max);
println(typeof(kernel))
#
# %% Initial measure
N = length(px)*length(py)
a0 = [1.]
x0 = [[-40., -pi/4]]
# Noise
#srand(1);
w0=randn(N);
sigma_noise=0.0;
# Load operator Phi
op=blasso.setGaussLineOperator(kernel,a0,x0,sigma_noise*w0);
image = zeros((length(px),length(py)))
for i in 1:length(a0)
	image += a0[i]*reshape(op.phi(x0[i]), (length(px),length(py)))
end
image += reshape(sigma_noise*w0, (length(px),length(py)))
pIm = heatmap(image, aspect_ratio=1)

# %% Compute the Radon Transform
angles = range(0, pi,200) |> collect

radon_t = radon(image, angles);
pradon = heatmap(radon_t)
plot(pIm, pradon)
