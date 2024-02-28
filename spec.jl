
import Pkg
Pkg.activate(".")

push!(LOAD_PATH,"./src/");

using Plots
using blasso, sfw, certificate, toolbox

# %Model constants
σ=.01;
Dpt= 1. /512;
pt=range(0., 1.,step=Dpt) |> collect;
N = length(pt)
pω=0.:N-1 |> collect;
Dpω=1.;
# Bounds of domain
θmin = -1.7
θmax = 1.7
# Option solver
options=sfw.sfw_options();
# Load kernel attributes
kernel=blasso.setSpecKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
println(typeof(kernel))
# println(kernel.grid)
#
# Initial measure
a_0=[1., 5.];
x_0=[[100., atan(-100)], [300., 0]]
# Noise
#srand(1);
w0=randn(N^2);
sigma=.005;
#print(typeof(w0*sigma))
# Load operator Phi
op=blasso.setSpecOperator(kernel, a_0,x_0,sigma*w0);
image = reshape(a_0[1] * op.phi(x_0[1]), (N,N))
image .+= reshape(a_0[2] * op.phi(x_0[2]), (N,N))

lambda=0.01;
# Load objective function
fobj=blasso.setfobj(op,lambda);
println(typeof(fobj))

result=sfw.sfw4blasso(fobj,kernel,op,options); # Solve problem
