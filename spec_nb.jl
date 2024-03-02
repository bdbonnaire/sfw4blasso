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

# ╔═╡ b4aea33e-6012-44b5-90ee-960e476382bd
begin
	# Initial measure
	a_0=[1.];
	x_0=[[100, pi/2 * .8]]
	# Noise
	#srand(1);
	w0=randn(N^2);
	sigma=.00;
	#print(typeof(w0*sigma))
	# Load operator Phi
	op=blasso.setSpecOperator(kernel, a_0,x_0,sigma*w0);
	image = zeros(N,N)
	for i in 1:length(a_0)
		image += reshape(a_0[i] * op.phi(x_0[i]), (N,N))
	end
	#image .+= reshape(sigma.*w0, (N,N))
	heatmap(image)
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
	lambda=0.01;
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
sfw.show_result(result, options)

# ╔═╡ 959b2184-787b-465e-9d1c-178510d9ea21
println(x_0)

# ╔═╡ 9f87f847-e175-4029-8870-eeeba7b6cebd
function plotSpikes2D(x0,a0,result, op)
	nb_rec_spikes = length(result.u) ÷ 3
	rec_amps = result.u[1:nb_rec_spikes]
	rec_diracs = result.u[nb_rec_spikes+1:end]
	rec_diracs = reshape(rec_diracs, (nb_rec_spikes,2))
	x0 = stack(x0)
	blasso.plotobservation(op)
	scatter!(x0[1,:], x0[2,:], label="Original", color=RGBA(1.,1.,1,0.5), marker=:circle, markerstroke=false, markersize=10)
	scatter!(rec_diracs[:,1], rec_diracs[:,2], label="Recovered Spikes", color=:red, marker=:circle, markerstrokecolor=false, markersize=4)
end

# ╔═╡ Cell order:
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═b4aea33e-6012-44b5-90ee-960e476382bd
# ╠═c468d81f-bfbc-4934-8518-58efcc551f72
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═67884e0d-db4a-4a6a-ace9-ec88efe65d14
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╠═959b2184-787b-465e-9d1c-178510d9ea21
# ╠═9f87f847-e175-4029-8870-eeeba7b6cebd
