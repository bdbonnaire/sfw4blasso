using Random, Distributions
using LaTeXStrings
	#=
"""
`main_alg(x0, a0, sigma, sigma_noise, lambda, [N, angle_min, angle_max]`

Convenience method to run the algorithm.

# Arguments
- `x0, a0` : original image spikes and amplitudes respectively
	x0 is of the form, [freq c; freq c; ...]
- `sigma` : relative width of the spectrogram's Gaussian window
- `sigma_noise` : standard deviation of Gaussian noise
- `lambda` : blasso parameter
- `N` : Amount of samples for the chirp signal (default is 256)
- `angle_min, angle_max` : parameter space bounds (default is ±π/2*.9)

# Return
- results : blasso result struct
- Plots obj : Original Image
- Plots obj : Recovered Lines
"""
	=#
function main_alg(x0::Array{Float64,2}, a0::Array{Float64}, sigma::Float64, sigma_noise::Float64, lambda::Float64, N::Int64=256, angle_min::Float64=-π/2*.9, angle_max::Float64=π/2*.9)
	####################### Main Algorithm #################################
	#================= Creating the spectrogram ===========================#
	freqs = x0[:,1]
	offsets = x0[:,2]
	sig = py"lin_chirp"(freqs[1], c=offsets[1] , N=N)[1]
	for i in 2:length(freqs)
		sig += py"lin_chirp"(freqs[i], c=offsets[i], N=N)[1]
	
	# Adding white Gaussian noise
	dist = Normal(0, sigma_noise)
	sig += rand(dist, ComplexF64, N)

	spec = py"gauss_spectrogram"(sig, sigma)

	# =================== sfw4blasso ==================================#
	# Setting up the kernel 
	Dpt= 1. /(N-1);
	pt=range(0., 1.,step=Dpt) |> collect;
	N = length(pt)
	pω=0.:N-1 |> collect;
	Dpω=1.;
	kernel = blasso.setSpecKernel(pt,pω,Dpt,Dpω, sigma, angle_min, angle_max);
	# Load operator Phi
	operator = blasso.setSpecOperator(kernel, vec(spec_harm_lin));
	
	fobj = blasso.setfobj(operator,lambda);
	# sfw4blasso
	options = sfw.sfw_options(max_mainIter=2);
	# Computing the results
	result=sfw.sfw4blasso(fobj,kernel,op,options) 

	###################### Plotting #########################################
	function plotResult_Lines()
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

		
		p_lines = plot(sizes=(1000,1000))

		heatmap!(x,y,y0)
	
		# Line Plotting
		#=
		##Plot the ground truth 
		for i in 1:length(x0)
			local yt = N*tan( x0[i][2] ) *  x[2:end-1] .+ N*x0[i][1];
			plot!(x[1:end-1], yt, lw=5, c=:black, label="Ground Truth")
		end
		=#
		# Plot the recovered lines
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
			margin=(0,:px),
			)

		return p_original, p_lines
	end
	p_original, p_lines = plotResult_Lines()

	#======================= Comparison =================================#
	
	a,x = blasso.decompAmpPos(result.u, d=2);
	a /= sigma*N^2
	a = sqrt.(a)
	x0_vecvec = [ x[i,:] for i in 1:length(x0) ]
	blasso.computeErrors(x0_vecvec, a0, x, a, operator)
	
	return result, p_original, p_lines
