import Pkg
Pkg.activate("..")
push!(LOAD_PATH,"../src")


using Plots 
using RadonKA
import Peaks
using blasso
using sfw

# %% Creating the Line

# Model constants
sigma=[1., 1.];
M = 32.;
MM = 65;
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
a0 = [1., 1.]
x0 = [[53., -pi/4], [-42., -pi/4]]
# Noise
#srand(1);
w0=randn(N);
sigma_noise=0.1;
# Load operator Phi
op=blasso.setGaussLineOperator(kernel,a0,x0,sigma_noise*w0);
image = zeros((length(px),length(py)))
for i in 1:length(a0)
	image += a0[i]*reshape(op.phi(x0[i]), (length(px),length(py)))
end
image += reshape(sigma_noise*w0, (length(px),length(py)))
pIm = heatmap(image, aspect_ratio=1)
plot!(x->M*cos(x)+M+1,x->M*sin(x)+M+1,0, 2pi; c=:black, lw=4)
# %% Compute the Radon Transform
angles = range(0, pi,200) |> collect

T = sqrt(2)M |> ceil
T = convert(Int64, T)
border_img = zeros((2T,2T))
border_img[(2T-MM)÷2 .+ (1:MM), (2T-MM)÷2 .+ (1:MM)] = image
heatmap(border_img)
lines_radon = -temp:temp
radon_t = radon(border_img, angles);
pradon = heatmap(radon_t)
# Computing peak of radon
peak = argmax(vec(radon_t))
peak_v = convert.(Float64, [peak % size(radon_t)[1], (peak ÷ size(radon_t)[1])+1])
scatter!([peak_v[2]],[peak_v[1]])
plot(pIm, pradon)


peak_real = copy(peak_v)
peak_real[2] *= -pi/200
#=
Discussion
==========
=#
