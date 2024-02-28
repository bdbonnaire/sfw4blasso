import Pkg
Pkg.activate(".")

push!(LOAD_PATH,"./src/");

using Plots
using blasso, sfw, certificate, toolbox

# Model constants
σ=.01;
Dpt= 1. /512;
pt=range(0., 1.,step=Dpt) |> collect;
N = length(pt)
pω=0.:N-1 |> collect;
Dpω=1.;
# Bounds of domain
θmin = -5π/12
θmax = 5π/12
# Option solver
options=sfw.sfw_options();
# Load kernel attributes
kernel=blasso.setSpecKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
println(typeof(kernel))
# println(kernel.grid)
## Initial measure
a_0=[1.];
x_0=[[100., atan(225)]]
# Noise
#srand(1);
w0=randn(N^2);
sigma=.0;
#print(typeof(w0*sigma))
# Load operator Phi
op=blasso.setSpecOperator(kernel, a_0,x_0,sigma*w0);
println(typeof(op))
# image = reshape(op.phi(x_0[1]), (N,N))
# heatmap(image)

lambda=0.01;
# Load objective function
fobj=blasso.setfobj(op,lambda);
println(typeof(fobj))

result=sfw.sfw4blasso(fobj,kernel,op,options); # Solve problem
