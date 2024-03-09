### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(".")
	Pkg.instanciate()
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
md"""
# Gaussian Lines Kernel
## Well-separated case
"""


# ╔═╡ 21f334a4-ef50-4e84-82c6-1d75a485d6b5
begin
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
end

# ╔═╡ b4aea33e-6012-44b5-90ee-960e476382bd
begin
	# Initial measure
	N = length(px)*length(py)
	a0=[1., 1., 1.];
	x0=[[0, -π/5], [-15, pi/16], [10, pi/6]];
	# Noise
	#srand(1);
	w0=randn(N);
	sigma_noise=0.31; # max phi = 1/sqrt(2*pi) multiply by the ratio 200/255 (noise/max xstar) to have the same amount of noise
	# Load operator Phi
	op=blasso.setGaussLineOperator(kernel,a0,x0,sigma_noise*w0);
	image = zeros((length(px),length(py)))
	for i in 1:length(a0)
		image += a0[i]*reshape(op.phi(x0[i]), (length(px),length(py)))
	end
	image += reshape(sigma_noise*w0, (length(px),length(py)))
	heatmap(image, aspect_ratio=1, cmap=:grays)
	#blasso.plotobservation(op)
end

# ╔═╡ c468d81f-bfbc-4934-8518-58efcc551f72
# ╠═╡ disabled = true
#=╠═╡
plotSpikes2D(x0,a0, result, op)
  ╠═╡ =#

# ╔═╡ 436b02fb-2b8b-4e66-93ca-e344ecd90df0
begin
	lambda=10.; # works to kill spurious spikes but threshold the amplitudes (need to perform a Least Square fitting to estimate the amplitudes)
	# Load objective function
	fobj=blasso.setfobj(op,lambda);
end

# ╔═╡ 67884e0d-db4a-4a6a-ace9-ec88efe65d14
#begin
#	etaV = certificate.computeEtaV(x0, sign.(a0), op)
#	#tt = collect(range(-1,1, 500))
#	#plot(tt, etaV(tt))
#end

# ╔═╡ 01ed0bc2-3c35-4d51-8d31-bb084b592879
result=sfw.sfw4blasso(fobj,kernel,op,options); # Solve problem

# ╔═╡ 3c8fb520-419c-4626-b42c-38c813385179
begin
	sfw.show_result(result, options)
end

# ╔═╡ fbb54e97-dc79-4bc3-bf8e-a33c10800762
begin
	a_est,x_est=blasso.decompAmpPos(result.u,d=op.dim);
	blasso.computeErrors(x0, a0, x_est, a_est);
end

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

# ╔═╡ c6b23a87-166e-4fc7-8cd9-a80b26912753
md"""
## Closed lines case
"""

# ╔═╡ 95b38abf-a574-4696-a8e8-4b71cc23a4da
begin
	# Initial measure
	a02=[1., 1.];
	x02=[[-1, -0.73], [1, -0.75]];
	# Noise
	#srand(1);
	w02=randn(N);
	sigma_noise2=0.031; # equivalent for max 255 to randn()*noiselevel with noiselevel=20
	# Load operator Phi
	op2=blasso.setGaussLineOperator(kernel,a02,x02,sigma_noise2*w02);
	image2 = zeros((length(px),length(py)))
	for i in 1:length(a02)
		image2 += a02[i]*reshape(op2.phi(x02[i]), (length(px),length(py)))
	end
	image2 += reshape(sigma_noise2*w02, (length(px),length(py)))
	heatmap(image2, aspect_ratio=1, cmap=:grays)
	#blasso.plotobservation(op)
end

# ╔═╡ 6141e714-46df-4f2a-ad8e-969374f7d7a6
begin
	lambda2=0.5;
	# Load objective function
	fobj2=blasso.setfobj(op2,lambda2);
end

# ╔═╡ 23a093f0-50f4-430a-abed-65119270e541
result2=sfw.sfw4blasso(fobj2,kernel,op2,options); # Solve problem

# ╔═╡ 6cb65afe-f7d4-48df-975b-2b25a5faac33
begin
	println("x0=$(x02)")
	sfw.show_result(result2, options)
end

# ╔═╡ 7c1bee1e-c0b0-4304-ae2b-96e195979e4f
begin
	a_est2,x_est2=blasso.decompAmpPos(result2.u,d=op2.dim);
	blasso.computeErrors(x02, a02, x_est2, a_est2);
end

# ╔═╡ 41f526a1-ee53-49f6-b308-9f051fc1a255
md"""
## Multiple lines case
"""

# ╔═╡ 76cb3f79-5a4a-4397-a254-2c676855ac26
begin
	# Initial measure
	a03=[60, 80, 255, 100, 180, 120, 240]/255;
	x03=[[15, -0.75], [25, -0.5], [2, -0.25], [7, 0.001], [-20, 0.3], [-5, 0.55], [-10, 0.75]];
	println(a03)
	# Noise
	#srand(1);
	w03=randn(N);
	sigma_noise3=0.031; # equivalent for max 255 to randn()*noiselevel with noiselevel=20
	# Load operator Phi
	op3=blasso.setGaussLineOperator(kernel,a03,x03,sigma_noise3*w03);
	image3 = zeros((length(px),length(py)))
	for i in 1:length(a03)
		image3 += a03[i]*reshape(op3.phi(x03[i]), (length(px),length(py)))
	end
	image3 += reshape(sigma_noise3*w03, (length(px),length(py)))
	heatmap(image3, aspect_ratio=1, cmap=:grays)
	#blasso.plotobservation(op)
end

# ╔═╡ a3e79a91-fd7d-4ac0-8519-4cc0e634a276
begin
	lambda3=1.;
	# Load objective function
	fobj3=blasso.setfobj(op3,lambda3);
end

# ╔═╡ ec4ec66b-7354-42e0-841f-de947dfd0f31
result3=sfw.sfw4blasso(fobj3,kernel,op3,options); # Solve problem

# ╔═╡ e3c5156a-0b35-40a8-939a-749626981446
# ╠═╡ disabled = true
#=╠═╡
begin
	a_est,x_est=blasso.decompAmpPos(result3.u,d=op3.dim);
	println(x_est[7]);
	a_est = a_est[1:end-1];
	x_est = x_est[1:end-1];
	op3.computeErrors(x03, a03, x_est, a_est);
	# sfw.show_result(result3, options)
end
  ╠═╡ =#

# ╔═╡ 6f4a51c3-9447-4c4f-a80b-2d98d3d51b73
begin
	a_est3,x_est3=blasso.decompAmpPos(result3.u,d=op3.dim);
	blasso.computeErrors(x03, a03, x_est3, a_est3);
end

# ╔═╡ Cell order:
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╠═5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═b4aea33e-6012-44b5-90ee-960e476382bd
# ╠═c468d81f-bfbc-4934-8518-58efcc551f72
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═67884e0d-db4a-4a6a-ace9-ec88efe65d14
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╠═fbb54e97-dc79-4bc3-bf8e-a33c10800762
# ╟─9f87f847-e175-4029-8870-eeeba7b6cebd
# ╠═c6b23a87-166e-4fc7-8cd9-a80b26912753
# ╠═95b38abf-a574-4696-a8e8-4b71cc23a4da
# ╠═6141e714-46df-4f2a-ad8e-969374f7d7a6
# ╠═23a093f0-50f4-430a-abed-65119270e541
# ╠═6cb65afe-f7d4-48df-975b-2b25a5faac33
# ╠═7c1bee1e-c0b0-4304-ae2b-96e195979e4f
# ╠═41f526a1-ee53-49f6-b308-9f051fc1a255
# ╠═76cb3f79-5a4a-4397-a254-2c676855ac26
# ╠═a3e79a91-fd7d-4ac0-8519-4cc0e634a276
# ╠═ec4ec66b-7354-42e0-841f-de947dfd0f31
# ╠═e3c5156a-0b35-40a8-939a-749626981446
# ╠═6f4a51c3-9447-4c4f-a80b-2d98d3d51b73
