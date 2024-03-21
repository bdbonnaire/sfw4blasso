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
	push!(LOAD_PATH,"../src/");

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

# ╔═╡ 1aa4ca75-8629-4132-8790-bb610e90c86b
begin
	using LaTeXStrings
	function plotResult_Lines(x0, y0, result, kernel, op)
		rec_amps, rec_diracs = blasso.decompAmpPos(result.u, d=2)
		max_amp = maximum(rec_amps)
		x = kernel.pt
		y = kernel.pω
		Npx = kernel.Npt
		Npy = kernel.Npω

		p_original = heatmap(y0,
			legend=:none, 
			cbar=:none, 
			framestyle=:none)
		plot!(sizes=(1000,1000))

		
		p_lines = plot()
		plot!(sizes=(1000,1000))

		heatmap!(x,y,y0)
	
		# Line Plotting
		#=
		##Plot the ground truth 
		for i in 1:length(x0)
			local yt = N*tan( x0[i][2] ) *  x[2:end-1] .+ N*x0[i][1];
			plot!(x[1:end-1], yt, lw=5, c=:black, label="Ground Truth")
		end
		=#
		# Plot the reocovered lines
		for i in 1:length(rec_amps)
			local y = N*tan( rec_diracs[i][2] ) *  x[2:end-1] .+ N*rec_diracs[i][1];
	
			local color = RGBA(1.,0.,0.,
				max(rec_amps[i]/max_amp,.4))
			plot!(x[2:end-1], y, lw=3, c=color, label="Recovered")
		end
		plot!(ylim=[y[1], y[end]],
			legend=:none, 
			cbar=:none, 
			framestyle=:none, 
			margin=(0,:px),)
			#title="Lines",
			#titlefontsize=22)

		#=
		## Parameter Space Plot
		p_parameterSpace = plot();
		x0_s = stack(x0)
		rec_dirac_s = stack(rec_diracs)
		scatter!(x0_s[1,:], x0_s[2,:], 
			markersize=15, 
			c=:black, 
			label="Ground Truth")	
		scatter!(rec_dirac_s[1,:], rec_dirac_s[2,:],
			markersize=6,
			c=RGBA.(1.,0.,0.,max.(rec_amps./max_amp,0.5)),
			label="Recovered")
		
		plot!(title="Spikes in the Parameter Space",
			titlefontsize=20,
			xlabel=L"$a$",
			ylabel=L"$\theta$",
			ylimit=[-pi/2, pi/2],
			#yticks=((-4:4)*pi/8,[L"-\frac{\pi}{2}", L"-\frac{3\pi}{8}", L"-\frac{\pi}{4}", L"-\frac{\pi}{8}", L"0", L"\frac{\pi}{8}",L"\frac{\pi}{4}",L"\frac{3\pi}{8}",L"\frac{\pi}{2}"]),
			yticks=((-2:2)*pi/4,[L"-\frac{\pi}{2}", L"-\frac{\pi}{4}", L"0",L"\frac{\pi}{4}",L"\frac{\pi}{2}"]),
			labelfontsize=18,
			xtickfontsize=14,
			ytickfontsize=18,
			yguidefontrotation=.9,
			legendfontsize=12,
			minorgrid=true,
			minorticks=2,
			framestyle=:box)
	
		plot(p_lines, p_parameterSpace)
		=#
		return p_original, p_lines
	end
end

# ╔═╡ 35e2ddf8-08da-40c9-9ec8-f6dc1e6b8d1c
pushfirst!(pyimport("sys")."path", "")

# ╔═╡ 81bdf362-fc97-499a-bb20-addabfe586ce
md"## Noiseless Case"

# ╔═╡ 9c7084c4-e7cb-45db-a762-15b821ed7263
begin
	pyplot()
	py"""
	from py_lib.signals import lin_chirp
	from py_lib.spectrogram import gauss_spectrogram
	
	import numpy as np
	#
	# Two linear chirps
	freq = 200
	freq_lin = 100
	a = -65
	N = 256
	sig1,_ = lin_chirp(freq, N=N)
	sig2, t = lin_chirp(freq_lin ,c=a, N=N)
	sig = sig1+sig2
	"""
	x0 = [[py"freq/N", 0], [py"freq_lin/N", atan(py"a/N")]]
	a0 = [1.,1.]
	spec_harm_lin = py"gauss_spectrogram"(py"sig", σ)
	p_spec = heatmap(abs.(spec_harm_lin), cbar=:none, framestyle=:none, margin=(0,:px) )
	savefig(p_spec, "spec.pdf")
end

# ╔═╡ d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
begin
	# Creating the spectrograms
	# Load operator Phi
	op=blasso.setSpecOperator(kernel, vec(spec_harm_lin));
	println(typeof(op))
end

# ╔═╡ d2c8ab19-ef76-42df-b5a0-88c01d15b34f
begin
	struct res
		λ
		u
		error_x
		error_a
		right_amount
	end
	function test_lambda(a0, x0, range_lambda, fobj, kernel, operator, options)
		N=length(x0) ;
		results = Array{res}(undef, length(range_lambda))
		for j in 1:length(range_lambda)
			λ = range_lambda[j]
			# Load objective function
			fobj=blasso.setfobj(op,λ);
			result=sfw.sfw4blasso(fobj,kernel,operator,options); # Solve problem
			a, x = blasso.decompAmpPos(result.u, d=2);
			M=min(N, length(a))
			#Compute error
			error_x = 0
			error_a = 0
			for i in 1:M		
				error_x += norm(x[i] .- x0[i]);
				error_a += norm(a[i] - a0[i]);
			end
			error_x /= M;
			error_a /= M;
			
			right_amount = (length(x) == N)
			results[j] = res(λ, result.u, error_x, error_a, right_amount)	
		end
		return results
	end
end

# ╔═╡ 436b02fb-2b8b-4e66-93ca-e344ecd90df0
begin
	lambda=10.;
	# Load objective function
	fobj=blasso.setfobj(op,lambda);
end

# ╔═╡ 01ed0bc2-3c35-4d51-8d31-bb084b592879
result=sfw.sfw4blasso(fobj,kernel,op,options) # Solve problem

# ╔═╡ 8c884d2d-2ae6-4ddb-9864-5d76e18035fd
let
a,b = plotResult_Lines(x0, spec_harm_lin, result, kernel, op)
plot(a); #savefig("figures/spec/spec.png")
plot(b);# savefig("figures/spec/spec_linesOnTopOnlyRed.png")
end

# ╔═╡ 3c8fb520-419c-4626-b42c-38c813385179
begin
	println("Original parameters = $x0")
	sfw.show_result(result, options)
end

# ╔═╡ da895779-bf60-4de4-a653-a9365c723b8c
typeof(op) == blasso.operator_spec_lchirp

# ╔═╡ 18d98400-3e2b-42b1-b07c-c78e46e0f0c2
let
	a,x = blasso.decompAmpPos(result.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	blasso.computeErrors(x0,a0, x, a, op)
end

# ╔═╡ 9081f8c7-9938-4b58-aa63-fa4d1562ebda
# ╠═╡ disabled = true
#=╠═╡
begin
	lambdas = range(0.01, 1, 8)
	results_λ = test_lambda([1.,1.],x0,lambdas, fobj, kernel, op, options)
end
  ╠═╡ =#

# ╔═╡ 4fa6816c-ca4d-4153-9cda-cebbe136a297
#=╠═╡
begin
	for i in 1:length(lambdas)
		println(results_λ[i].error_x)
	end
end
  ╠═╡ =#

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
	a = -65
	N = 256
	sig1,_ = lin_chirp(freq, N=N)
	sig2, t = lin_chirp(freq_lin ,c=a, N=N)
	sig = sig1+sig2
	
	# adding noise
	sigma_noise = 1.
	rng = np.random.default_rng()
	noise = rng.normal(0, sigma_noise, N) + rng.normal(0, sigma_noise, N)*1j
	sig_noisy = sig + noise
	"""
	# computing spectrogram
	spec_harm_lin_noisy = py"gauss_spectrogram"(py"sig_noisy", σ)
	
	Pspec_noisy = heatmap(abs.(spec_harm_lin_noisy), cbar=:none, framestyle=:none, margin=(0,:px) )
	#savefig(Pspec_noisy, "spec_noisy.png")
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

# ╔═╡ a5eb30b7-b5d9-45fb-98df-df12731a26f8
let
a,b = plotResult_Lines(x0, spec_harm_lin_noisy, result_noisy, kernel, op_noisy)
plot(a); #savefig("figures/spec/spec_noisy.png")
plot(b); #savefig("figures/spec/spec_noisy_linesOnTopOnlyRed.png")
end

# ╔═╡ 50facba1-8795-4a84-b480-6a27cf5b5d57
# ╠═╡ disabled = true
#=╠═╡
let
	a,x = blasso.decompAmpPos(result_noisy.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	blasso.computeErrors(x0,a0, x, a, op_noisy)
end
  ╠═╡ =#

# ╔═╡ 26fcfbb7-a611-4a0d-a6ed-b303e9ffad4c
let
	a,x = blasso.decompAmpPos(result_noisy.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	open("errors/spec_noisy","a") do out
		redirect_stdout(out) do		
			blasso.computeErrors(x0,a0, x, a, op_noisy)
		end
	end
end


# ╔═╡ 0270d7ad-4d6b-47e8-b5a5-ba554551161e
# ╠═╡ disabled = true
#=╠═╡
begin
	lambdas_noisy = range(0.01, 1, 8)
	results_λ_noisy = test_lambda([1.,1.],x0,lambdas_noisy, fobj_noisy, kernel, op_noisy, options)
end
  ╠═╡ =#

# ╔═╡ 43699165-4a48-4aeb-ba21-4ebf240362d9
#=╠═╡
for i in 1:8
	println("Mean error in spike location $(results_λ_noisy[i].error_x)")
end
  ╠═╡ =#

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
	sigma_noise = 0.2
	rng = np.random.default_rng()
	noise = rng.normal(0, sigma_noise, N) + rng.normal(0, sigma_noise, N)*1j
	sig_noisy = sig + noise
	"""
	# computing spectrogram
	spec_harm_lin_interf = py"gauss_spectrogram"(py"sig_noisy", σ)
	x0_interf = [[py"freq1/N",0], [py"(freq1-df)/N", 0]]
	
	Pspec_interf = heatmap(abs.(spec_harm_lin_interf), cbar=:none, framestyle=:none, margin=(0,:px) )
	#savefig(Pspec_interf, "spec_interf.pdf")
end

# ╔═╡ 4cc6a77e-6356-4557-b396-5c2ffd22b78e
begin	
	op_interf=blasso.setSpecOperator(kernel, vec(spec_harm_lin_interf));
	lambda_interf=0.01;
	# Load objective function
	fobj_interf=blasso.setfobj(op_interf,lambda_interf);
end

# ╔═╡ 57eec84a-7ede-48a8-9708-8ba21a82272e
# ╠═╡ disabled = true
#=╠═╡
let
	a,x = blasso.decompAmpPos(result.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	open("errors/spec_interf","a") do out
		redirect_stdout(out) do		
			blasso.computeErrors(x0_interf,a0, x, a, op_interf)
		end
	end
end
  ╠═╡ =#

# ╔═╡ bffd0638-e81d-4904-b7b6-67210dc0721b
result_interf=sfw.sfw4blasso(fobj_interf,kernel,op_interf,options) # Solve problem

# ╔═╡ 1e30b4fd-8ee0-4c5f-a7a0-a290c3d6d3b6
let
a,b = plotResult_Lines(x0_interf, spec_harm_lin_interf, result_interf, kernel, op_interf)
plot(a); #savefig("figures/spec/spec_interf.png")
plot(b); #savefig("figures/spec/spec_interf_linesOnTopOnlyRed.png")
end

# ╔═╡ b3ed07d7-34d1-4a54-b796-da31b79270af
# ╠═╡ disabled = true
#=╠═╡
let
	a,x = blasso.decompAmpPos(result_interf.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	blasso.computeErrors(x0_interf,a0, x, a, op)
end
  ╠═╡ =#

# ╔═╡ 30f1fdfa-5030-4a97-85aa-0e57c7dc46e1
let
	a,x = blasso.decompAmpPos(result_interf.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	open("errors/spec_interf","a") do out
		redirect_stdout(out) do		
			blasso.computeErrors(x0,a0, x, a, op_interf)
		end
	end
end

# ╔═╡ a57b29f7-f3bf-4d50-bcd7-27e414df9822
begin
	println("Original parameters = $x0_interf")
	sfw.show_result(result_interf, options)
end

# ╔═╡ a9d43938-0fd1-4153-a09c-8cadab81df71
# ╠═╡ disabled = true
#=╠═╡
begin
	lambdas_interf = range(0.01, 1, 8)
	results_λ_interf = test_lambda([1.,1.],x0_interf,lambdas_interf, fobj_interf, kernel, op_interf, options)
end
  ╠═╡ =#

# ╔═╡ 6dd8c677-fed7-4e20-a896-e2cc189743ec
#=╠═╡
for i in 1:8
	println("Mean error in spike location $(results_λ_interf[i].error_x)")
end
  ╠═╡ =#

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
	sigma_noise = 0.2
	rng = np.random.default_rng()
	noise = rng.normal(0, sigma_noise, N) + rng.normal(0, sigma_noise, N)*1j
	sig_noisy = sig + noise
	"""
	# computing spectrogram
	spec_harm_lin_crossing = py"gauss_spectrogram"(py"sig_noisy", σ)
	x0_crossing = [[py"freq1/N", -atan(py"2*df/N")], [py"freq2/N", atan(py"2*df/N")]]
	
	Pspec_crossing = heatmap(abs.(spec_harm_lin_crossing), cbar=:none, framestyle=:none, margin=(0,:px) )
	#savefig(Pspec_crossing, "spec_crossing.pdf")
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

# ╔═╡ 5d5ba14a-acae-4437-a216-27394c8a068a
# ╠═╡ disabled = true
#=╠═╡
begin
	abc = plotResult_Lines(x0_crossing, spec_harm_lin_crossing, result_crossing, kernel, op_crossing)
	savefig(abc, "figures/spec_crossing_linesOnTop.pdf")
end
  ╠═╡ =#

# ╔═╡ 66baf60d-1870-4c0b-b7f0-9d40b69a4638
let
a,b = plotResult_Lines(x0_crossing, spec_harm_lin_crossing, result_crossing, kernel, op_crossing)
plot(a); #savefig("figures/spec/spec_crossing.png")
plot(b); #savefig("figures/spec/spec_crossing_linesOnTopOnlyRed.png")
end

# ╔═╡ ed89bbc7-1043-4062-8d72-0a300d8c6370
# ╠═╡ disabled = true
#=╠═╡
let
	a,x = blasso.decompAmpPos(result_crossing.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	blasso.computeErrors(x0_crossing,a0, x, a, op)
end
  ╠═╡ =#

# ╔═╡ c6afcd44-11c0-4df1-88e5-4309294516ee
let
	a,x = blasso.decompAmpPos(result_crossing.u, d=2)
	a /= σ*N^2
	a = sqrt.(a)
	open("errors/spec_crossing","a") do out
		redirect_stdout(out) do		
			blasso.computeErrors(x0,a0, x, a, op_crossing)
		end
	end
end

# ╔═╡ e375d490-c8f1-42fd-ae8c-fe881a0bd289
# ╠═╡ disabled = true
#=╠═╡
begin
	a_est_crossing,x_est_crossing=blasso.decompAmpPos(result_crossing.u,d=op_crossing.dim);
	open("errors/spec_crossing","a") do out
		redirect_stdout(out) do		
			blasso.computeErrors(x0_crossing, [0.,0.], x_est_crossing, a_est_crossing, op_crossing);
		end
	end
end
  ╠═╡ =#

# ╔═╡ c877e2d4-3818-4316-aba0-c955e1af7fda
begin
	println("Original parameters = $x0_crossing")
	sfw.show_result(result_crossing, options)
end

# ╔═╡ 4e8dc31e-dfbf-48e1-9578-44bcab1b7736
# ╠═╡ disabled = true
#=╠═╡
begin
	lambdas_crossing = range(0.01, 1, 8)
	results_λ_crossing = test_lambda([1.,1.],x0,lambdas_crossing, fobj_crossing, kernel, op_crossing, options)
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═c13a86de-cb38-11ee-3890-c93e2ad0f39a
# ╠═9e13dfd5-078d-49bb-827e-97575a6a42df
# ╠═bb6f403c-0897-4903-be58-8cd320f83d17
# ╠═d2c8ab19-ef76-42df-b5a0-88c01d15b34f
# ╟─5dcf6210-5e2d-4c74-854e-5617749d8b8c
# ╟─648eff19-26fa-4d90-8968-daaacda95974
# ╠═21f334a4-ef50-4e84-82c6-1d75a485d6b5
# ╠═35e2ddf8-08da-40c9-9ec8-f6dc1e6b8d1c
# ╟─81bdf362-fc97-499a-bb20-addabfe586ce
# ╠═9c7084c4-e7cb-45db-a762-15b821ed7263
# ╠═d71103a1-8e24-48b5-b6cd-9e9cf7a734a3
# ╠═436b02fb-2b8b-4e66-93ca-e344ecd90df0
# ╠═8c884d2d-2ae6-4ddb-9864-5d76e18035fd
# ╠═01ed0bc2-3c35-4d51-8d31-bb084b592879
# ╟─1aa4ca75-8629-4132-8790-bb610e90c86b
# ╠═3c8fb520-419c-4626-b42c-38c813385179
# ╠═da895779-bf60-4de4-a653-a9365c723b8c
# ╠═18d98400-3e2b-42b1-b07c-c78e46e0f0c2
# ╠═9081f8c7-9938-4b58-aa63-fa4d1562ebda
# ╠═4fa6816c-ca4d-4153-9cda-cebbe136a297
# ╟─0dec2a4e-dd48-4ad1-9f89-54d63877129b
# ╠═7dea5cc0-9a54-4b55-90a5-9b1d3fc1fe5f
# ╠═5092307e-6c23-46f9-b119-6ae8ce3e0604
# ╠═a5eb30b7-b5d9-45fb-98df-df12731a26f8
# ╠═83877111-4e59-4c6d-830a-ecf9aea28bda
# ╠═50facba1-8795-4a84-b480-6a27cf5b5d57
# ╠═26fcfbb7-a611-4a0d-a6ed-b303e9ffad4c
# ╠═0270d7ad-4d6b-47e8-b5a5-ba554551161e
# ╠═43699165-4a48-4aeb-ba21-4ebf240362d9
# ╠═2f47da1b-b768-4f51-b83f-e6f67a231e2b
# ╠═dc8263c3-8e41-4918-bf70-f69d265b6597
# ╠═4cc6a77e-6356-4557-b396-5c2ffd22b78e
# ╠═1e30b4fd-8ee0-4c5f-a7a0-a290c3d6d3b6
# ╠═57eec84a-7ede-48a8-9708-8ba21a82272e
# ╠═b3ed07d7-34d1-4a54-b796-da31b79270af
# ╠═30f1fdfa-5030-4a97-85aa-0e57c7dc46e1
# ╠═bffd0638-e81d-4904-b7b6-67210dc0721b
# ╠═a57b29f7-f3bf-4d50-bcd7-27e414df9822
# ╠═a9d43938-0fd1-4153-a09c-8cadab81df71
# ╠═6dd8c677-fed7-4e20-a896-e2cc189743ec
# ╟─34a3bc8e-b75e-4797-9db3-db47711f34a0
# ╠═caad4e0a-3b67-4fd6-a6be-4e9045de4c73
# ╠═9220d643-eda8-4464-ad2d-54c7ee7a82e3
# ╠═5d5ba14a-acae-4437-a216-27394c8a068a
# ╠═66baf60d-1870-4c0b-b7f0-9d40b69a4638
# ╠═44184a53-9acc-449e-9fe3-0e3c23e796c5
# ╠═ed89bbc7-1043-4062-8d72-0a300d8c6370
# ╠═c6afcd44-11c0-4df1-88e5-4309294516ee
# ╠═e375d490-c8f1-42fd-ae8c-fe881a0bd289
# ╠═c877e2d4-3818-4316-aba0-c955e1af7fda
# ╠═4e8dc31e-dfbf-48e1-9578-44bcab1b7736