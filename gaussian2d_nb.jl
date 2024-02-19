### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(".")
	Pkg.add("SpecialFunctions")
	Pkg.add("Revise")
	using LinearAlgebra, Plots
end;

# ╔═╡ 9e13dfd5-078d-49bb-827e-97575a6a42df
	push!(LOAD_PATH,"./src/");

# ╔═╡ bb6f403c-0897-4903-be58-8cd320f83d17
begin 
	using Revise
	using blasso, sfw, certificate, toolbox
end

# ╔═╡ 5dcf6210-5e2d-4c74-854e-5617749d8b8c
md"# Gaussian 2D Kernel"

# ╔═╡ 21f334a4-ef50-4e84-82c6-1d75a485d6b5
begin
	# Model constants
	sigmax=.01;
	sigmay=.01;
	Dpx=.05;
	Dpy=.05;
	px=range(0., 1.,step=Dpx);
	px=collect(px);
	py=px;
	# Bounds of domain
	bounds=[[0.,0.], [1.0,1.0]];
	# Option solver
	options=sfw.sfw_options();
	# Load kernel attributes
	kernel=blasso.setKernel(px,py,Dpx,Dpy,sigmax,sigmay,bounds);
	println(typeof(kernel))
end

# ╔═╡ b4aea33e-6012-44b5-90ee-960e476382bd
begin
	# Initial measure
	N = length(px)*length(py)
	a0=[.8,.5];
	x0=[[.7,.7], [.2, .3]];
	# Noise
	#srand(1);
	w0=zeros(N);#randn(N);
	sigma=.01;
	# Load operator Phi
	op=blasso.setoperator(kernel,a0,x0,sigma*w0);
	blasso.plotobservation(op)
end

# ╔═╡ 436b02fb-2b8b-4e66-93ca-e344ecd90df0
begin
	lambda=.5;
	# Load objective function
	fobj=blasso.setfobj(op,lambda);
end

# ╔═╡ 67884e0d-db4a-4a6a-ace9-ec88efe65d14
begin
	etaV = certificate.computeEtaV(x0, sign.(a0), op)
	#tt = collect(range(-1,1, 500))
	#plot(tt, etaV(tt))
end

# ╔═╡ 01ed0bc2-3c35-4d51-8d31-bb084b592879
result=sfw.sfw4blasso(fobj,kernel,op,options); # Solve problem

# ╔═╡ 3c8fb520-419c-4626-b42c-38c813385179
sfw.show_result(result, options)

# ╔═╡ 9f87f847-e175-4029-8870-eeeba7b6cebd
function plotSpikes(x0,a0,result)
	rec_amps = result.u[1:length(x0)]
	rec_diracs = result.u[length(x0)+1:end]
	plot(rec_diracs, rec_amps, line=:stem, label="Recovered Spikes", ls=:solid, lw=2, color=:red, marker=:circle, markersize=4)
	plot!(x0, a0, line=:stem, label="Original", ls=:dash, color=:black, marker=:circle, markersize=2)
end

# ╔═╡ Cell order:
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═b4aea33e-6012-44b5-90ee-960e476382bd
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═67884e0d-db4a-4a6a-ace9-ec88efe65d14
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╠═9f87f847-e175-4029-8870-eeeba7b6cebd
