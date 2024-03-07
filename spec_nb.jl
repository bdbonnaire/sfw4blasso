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
	using Revise
	using PyCall
	using blasso, sfw, certificate, toolbox
end

# ╔═╡ 5dcf6210-5e2d-4c74-854e-5617749d8b8c
md"# Spectrogram Kernel"

# ╔═╡ 648eff19-26fa-4d90-8968-daaacda95974
md"## Setting up the kernel"

# ╔═╡ 21f334a4-ef50-4e84-82c6-1d75a485d6b5
begin
	# Model constants
	N=256
	σ=.02;
	Dpt= 1. /(N-1);
	pt=range(0., 1.,step=Dpt) |> collect;
	N = length(pt)
	pω=0.:N-1 |> collect;
	Dpω=1.;
	# Bounds of domain
	θmin = -π/2 * .9
	θmax = π/2 *.9
	# Option solver
	options=sfw.sfw_options(max_mainIter=2);
	# Load kernel attributes
	kernel=blasso.setSpecKernel(pt,pω,Dpt,Dpω,σ, θmin, θmax);
	println(typeof(kernel))
end

# ╔═╡ 35e2ddf8-08da-40c9-9ec8-f6dc1e6b8d1c
pushfirst!(pyimport("sys")."path", "")

# ╔═╡ 81bdf362-fc97-499a-bb20-addabfe586ce
md"## Noiseless Case"

# ╔═╡ 9c7084c4-e7cb-45db-a762-15b821ed7263
begin
	py"""
	from py_lib.signals import lin_chirp
	from py_lib.spectrogram import gauss_spectrogram
	
	import numpy as np
	#
	# Two linear chirps
	freq = 200
	freq_lin = 100
	a = -80
	N = 256
	sig1,_ = lin_chirp(freq, N=N)
	sig2, t = lin_chirp(freq_lin ,c=a, N=N)
	sig = sig1+sig2
	"""
	x0 = [[py"freq/N", 0], [py"freq_lin/N", atan(py"a/N")]]
	spec_harm_lin = py"gauss_spectrogram"(py"sig", σ)
	heatmap(abs.(spec_harm_lin), title="Spectrogram");
end

# ╔═╡ d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
begin
	# Creating the spectrograms
	# Load operator Phi
	op=blasso.setSpecOperator(kernel, vec(spec_harm_lin));
	println(typeof(op))
end

# ╔═╡ 436b02fb-2b8b-4e66-93ca-e344ecd90df0
begin
	lambda=0.01;
	# Load objective function
	fobj=blasso.setfobj(op,lambda);
end

# ╔═╡ 01ed0bc2-3c35-4d51-8d31-bb084b592879
result=sfw.sfw4blasso(fobj,kernel,op,options) # Solve problem

# ╔═╡ 3c8fb520-419c-4626-b42c-38c813385179
begin
	println("Original parameters = $x0")
	sfw.show_result(result, options)
end

# ╔═╡ 0dec2a4e-dd48-4ad1-9f89-54d63877129b
md"## Noisy Case"

# ╔═╡ 7dea5cc0-9a54-4b55-90a5-9b1d3fc1fe5f
begin
	py"""
	from py_lib.signals import lin_chirp
	from py_lib.spectrogram import gauss_spectrogram
	
	import numpy as np
	#
	# Two linear chirps
	freq = 200
	freq_lin = 100
	a = -80
	N = 256
	sig1,_ = lin_chirp(freq, N=N)
	sig2, t = lin_chirp(freq_lin ,c=a, N=N)
	sig = sig1+sig2
	
	# adding noise
	sigma_noise = 0.2
	rng = np.random.default_rng()
	noise = rng.normal(0, sigma_noise, N) + rng.normal(0, sigma_noise, N)*1j
	sig_noisy = sig + noise
	"""
	# computing spectrogram
	spec_harm_lin_noisy = py"gauss_spectrogram"(py"sig_noisy", σ)
	
	Pspec_noisy = heatmap(abs.(spec_harm_lin_noisy), title="Noisy", cbar=false);
end

# ╔═╡ 5092307e-6c23-46f9-b119-6ae8ce3e0604
begin	
	op_noisy=blasso.setSpecOperator(kernel, vec(spec_harm_lin_noisy));
	lambda_noisy=0.01;
	# Load objective function
	fobj_noisy=blasso.setfobj(op_noisy,lambda_noisy);
end

# ╔═╡ 83877111-4e59-4c6d-830a-ecf9aea28bda
result_noisy=sfw.sfw4blasso(fobj_noisy,kernel,op_noisy,options) # Solve problem

# ╔═╡ 26fcfbb7-a611-4a0d-a6ed-b303e9ffad4c
begin
	println("Original parameters = $x0")
	sfw.show_result(result, options)
end

# ╔═╡ 2f47da1b-b768-4f51-b83f-e6f67a231e2b
md"## Interfering Case"

# ╔═╡ dc8263c3-8e41-4918-bf70-f69d265b6597
begin
	py"""
	from py_lib.signals import lin_chirp
	from py_lib.spectrogram import gauss_spectrogram
	
	import numpy as np
	#
	# Two linear chirps
	freq1 = 150
	df = 60
	N = 256
	sig1,_ = lin_chirp(freq1, N=N)
	sig2, t = lin_chirp(freq1 -df, N=N)
	sig = sig1+sig2
	
	# adding noise
	#sigma_noise = 0.2
	#rng = np.random.default_rng()
	#noise = rng.normal(0, sigma_noise, N) + rng.normal(0, sigma_noise, N)*1j
	#sig_noisy = sig + noise
	"""
	# computing spectrogram
	spec_harm_lin_interf = py"gauss_spectrogram"(py"sig", σ)
	x0_interf = [[py"freq1/N",0], [py"(freq1-df)/N", 0]]
	
	Pspec_interf = heatmap(abs.(spec_harm_lin_interf), title="Interferences", cbar=false);
end

# ╔═╡ 4cc6a77e-6356-4557-b396-5c2ffd22b78e
begin	
	op_interf=blasso.setSpecOperator(kernel, vec(spec_harm_lin_interf));
	lambda_interf=0.01;
	# Load objective function
	fobj_interf=blasso.setfobj(op_interf,lambda_interf);
end

# ╔═╡ bffd0638-e81d-4904-b7b6-67210dc0721b
result_interf=sfw.sfw4blasso(fobj_interf,kernel,op_interf,options) # Solve problem

# ╔═╡ a57b29f7-f3bf-4d50-bcd7-27e414df9822
begin
	println("Original parameters = $x0_interf")
	sfw.show_result(result_interf, options)
end

# ╔═╡ 34a3bc8e-b75e-4797-9db3-db47711f34a0
md"## Crossing case"

# ╔═╡ caad4e0a-3b67-4fd6-a6be-4e9045de4c73
begin
	py"""
	from py_lib.signals import lin_chirp
	from py_lib.spectrogram import gauss_spectrogram
	
	import numpy as np
	#
	# Two linear chirps
	N = 256
	df = 100
	freq1 = N//2 + df
	freq2 = N//2 - df
	sig1,_ = lin_chirp(freq1, c=-2*df, N=N)
	sig2, t = lin_chirp(freq2, c=2*df, N=N)
	sig = sig1+sig2
	
	# adding noise
	#sigma_noise = 0.2
	#rng = np.random.default_rng()
	#noise = rng.normal(0, sigma_noise, N) + rng.normal(0, sigma_noise, N)*1j
	#sig_noisy = sig + noise
	"""
	# computing spectrogram
	spec_harm_lin_crossing = py"gauss_spectrogram"(py"sig", σ)
	x0_crossing = [[py"freq1/N", -atan(py"2*df/N")], [py"freq2/N", atan(py"2*df/N")]]
	
	Pspec_crossing = heatmap(abs.(spec_harm_lin_crossing), title="Crossing Modes", cbar=false);
end

# ╔═╡ 9220d643-eda8-4464-ad2d-54c7ee7a82e3
begin	
	op_crossing=blasso.setSpecOperator(kernel, vec(spec_harm_lin_crossing));
	lambda_crossing=0.01;
	# Load objective function
	fobj_crossing=blasso.setfobj(op_crossing,lambda_crossing);
end

# ╔═╡ 44184a53-9acc-449e-9fe3-0e3c23e796c5
result_crossing=sfw.sfw4blasso(fobj_crossing,kernel,op_crossing,options) # Solve problem

# ╔═╡ c877e2d4-3818-4316-aba0-c955e1af7fda
begin
	println("Original parameters = $x0_crossing")
	sfw.show_result(result_crossing, options)
end

# ╔═╡ Cell order:
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╟─648eff19-26fa-4d90-8968-daaacda95974
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═35e2ddf8-08da-40c9-9ec8-f6dc1e6b8d1c
# ╟─81bdf362-fc97-499a-bb20-addabfe586ce
# ╠═9c7084c4-e7cb-45db-a762-15b821ed7263
# ╠═d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╟─0dec2a4e-dd48-4ad1-9f89-54d63877129b
# ╠═7dea5cc0-9a54-4b55-90a5-9b1d3fc1fe5f
# ╠═5092307e-6c23-46f9-b119-6ae8ce3e0604
# ╠═83877111-4e59-4c6d-830a-ecf9aea28bda
# ╠═26fcfbb7-a611-4a0d-a6ed-b303e9ffad4c
# ╠═2f47da1b-b768-4f51-b83f-e6f67a231e2b
# ╠═dc8263c3-8e41-4918-bf70-f69d265b6597
# ╠═4cc6a77e-6356-4557-b396-5c2ffd22b78e
# ╠═bffd0638-e81d-4904-b7b6-67210dc0721b
# ╠═a57b29f7-f3bf-4d50-bcd7-27e414df9822
# ╟─34a3bc8e-b75e-4797-9db3-db47711f34a0
# ╠═caad4e0a-3b67-4fd6-a6be-4e9045de4c73
# ╠═9220d643-eda8-4464-ad2d-54c7ee7a82e3
# ╠═44184a53-9acc-449e-9fe3-0e3c23e796c5
# ╠═c877e2d4-3818-4316-aba0-c955e1af7fda
