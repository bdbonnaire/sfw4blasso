### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(".")
	using LinearAlgebra, Plots
end;

# ╔═╡ 9e13dfd5-078d-49bb-827e-97575a6a42df
	push!(LOAD_PATH,"./src/");

# ╔═╡ bb6f403c-0897-4903-be58-8cd320f83d17
begin 
	#using Revise
	using blasso, sfw, certificate, toolbox
end

# ╔═╡ 5dcf6210-5e2d-4c74-854e-5617749d8b8c
md"# Spectrogram Kernel"

# ╔═╡ 21f334a4-ef50-4e84-82c6-1d75a485d6b5
begin
	# Model constants
	σ=.01;
	Dpt= 1. /512;
	pt=range(0., 1.,step=Dpt) |> collect;
	N = length(pt)
	pω=0.:N-1 |> collect;
	Dpω=1.;
	# Bounds of domain
	θmin = -π/2 * .9
	θmax = π/2 *.9
	# Option solver
	options=sfw.sfw_options();
	# Load kernel attributes
	kernel=blasso.setSpecKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
	println(typeof(kernel))
end

# ╔═╡ d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
begin
	# Initial measure
	a0=[1., 1.];
	x0=[[.1, pi/4], [0.7, -pi/6]]
	# Noise
	#srand(1);
	w0=randn(N^2);
	sigma=.01;
	#print(typeof(w0*sigma))
	# Load operator Phi
	op=blasso.setSpecOperator(kernel, a0,x0,sigma*w0);
	println(typeof(op))
end

# ╔═╡ 3d43a392-db29-4f4c-ba69-3d17d4978155
begin
	local image = zeros(N,N)
	for i in 1:length(a0)
		image += reshape(a0[i] * op.phi(x0[i]), (N,N))
	end
	image .+= reshape(sigma.*w0, (N,N))
	Plots.heatmap(image)
	#blasso.plotobservation(op)
end

# ╔═╡ c468d81f-bfbc-4934-8518-58efcc551f72
# ╠═╡ disabled = true
#=╠═╡
begin
	ar, xr = blasso.decompAmpPos(result.u, d=2)
	imageRes = reshape(ar[1] * op.phi(xr[1]), (N,N))
	imageRes .+= reshape(ar[2] * op.phi(xr[2]), (N,N))
	imageRes .+= reshape(sigma.*w0, (N,N))
	heatmap(imageRes)
end
  ╠═╡ =#

# ╔═╡ 436b02fb-2b8b-4e66-93ca-e344ecd90df0
begin
	lambda=1.;
	# Load objective function
	fobj=blasso.setfobj(op,lambda);
end

# ╔═╡ 67884e0d-db4a-4a6a-ace9-ec88efe65d14
# ╠═╡ disabled = true
#=╠═╡
#begin
#	etaV = certificate.computeEtaV(x0, sign.(a0), op)
#	#tt = collect(range(-1,1, 500))
#	#plot(tt, etaV(tt))
#end
  ╠═╡ =#

# ╔═╡ 01ed0bc2-3c35-4d51-8d31-bb084b592879
result=sfw.sfw4blasso(fobj,kernel,op,options) # Solve problem

# ╔═╡ 3c8fb520-419c-4626-b42c-38c813385179
begin
	println(x0)
	sfw.show_result(result, options)
end

# ╔═╡ 9614902e-f341-46c1-9cc9-16c3fbac29bb
begin
	a_est,x_est=blasso.decompAmpPos(result.u,d=op.dim);
	blasso.computeErrors(x0, a0, x_est, a_est);
end

# ╔═╡ Cell order:
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
# ╠═3d43a392-db29-4f4c-ba69-3d17d4978155
# ╠═c468d81f-bfbc-4934-8518-58efcc551f72
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═67884e0d-db4a-4a6a-ace9-ec88efe65d14
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╠═9614902e-f341-46c1-9cc9-16c3fbac29bb
