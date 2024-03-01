

import Pkg
Pkg.activate(".")

push!(LOAD_PATH,"./src/");

using LinearAlgebra, Plots
using blasso, sfw, certificate, toolbox

# %Model constants
σ=.01;
Dpt= 1. /512;
pt=range(0., 1.,step=Dpt) |> collect;
N = length(pt)
pω=(0.:N-1 )/N|> collect;
Dpω=1. /N;
# Bounds of domain
θmin = -π/2 * .9
θmax = π/2 * .9
# Option solver
options=sfw.sfw_options();
# Load kernel attributes
kernelT=blasso.setSpecTKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
#
# Initial measure
a_0=[1.];
x_0T=[[.5, 0]]
x_0=[[.5*512, 0]]
# Noise
#srand(1);
w0=randn(N^2);
sigma=.000;
#print(typeof(w0*sigma))
# Load operator Phi
opT=blasso.setSpecTOperator(kernelT, a_0,x_0T,sigma*w0);

lambda=0.0001;
# Load objective function
fobj=blasso.setfobj(opT,lambda);
println(typeof(fobj))

result=sfw.sfw4blasso(fobj,kernelT,opT,options); # Solve problem

