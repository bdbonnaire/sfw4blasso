mutable struct gaussianLines <: discrete
  dim::Int64
  pt::Array{Float64,1}
  pω::Array{Float64,1}
  p::Array{Array{Float64,1},1}
  Npt::Int64
  Npω::Int64
  Dpt::Float64
  Dpω::Float64

  nbpointsgrid::Array{Int64,1}
  grid::Array{Array{Float64,1},1}
  meshgrid::Array{Array{Float64, 1}, 2}

  σ::Float64
  bounds::Array{Array{Float64,1},1}
end

""" setspecKernel(pt, pω, Dpt, Dpω, σ, angle_min, angle_max)
Sets the kernel structure `spec_lchirp` corresponding to φ: X -> H in the sfw4blasso paper.

In our case, X=[η_min, η_max]x[θ_min, θ_max] where 
- θ_min, θ_max correspond to `angle_min`, `angle_max` and 
- η_min and η_max are computed from θ_min and θ_max.

# Args
- pt, pω : Array{Float64,1} 
	the time and frequency grid on H 
- Dpt, Dpω : Float64
	The spacing in `pt` and `pω`
- σ : Float64
	window width used to compute the spectrogram
- angle_min, angle_max : Float64
	min and max values for the angle. [angle_min, angle_max] must be included in [-π/2, π/2).
"""
function setGaussLinesKernel(pt::Array{Float64,1},pω::Array{Float64,1},Dpt::Float64,Dpω::Float64, σ::Float64, angle_min::Float64, angle_max::Float64)

  dim=2;
  # Sampling of the Kernel
  Npt,Npω=length(pt),length(pω);
  p=Array{Array{Float64,1}}(undef,Npt*Npω);
  for i in 1:Npω
	  for j in 1:Npt
		  p[(i-1)*Npt+j] = [pt[j],pω[i]]
	  end
  end

  # Sampling of the parameter space X
  #		TODO: Generalize. This is only verified when considering a signal sampled over 1s. Otherwise the found η might not correlate with frequencies
  ## Computing the bounds of X
  θ_min = angle_min
  θ_max = angle_max
  η_min = -tan(θ_max)
  η_max = 1 - tan(θ_min)
  bounds = [[η_min, θ_min], [η_max, θ_max]]
  println("ηmin = $η_min et ηmax = $η_max")
  println("θmin = $θ_min et θmax = $θ_max")
  ## Computing the grid
  freq_coeff = .004
  angle_coeff = .02
  # makes the number of parameter grid samples proportional to the span of [ηmin, ηmax] and [θmin, θmax] resp.
  nb_points_param_grid = [ Npω * abs(η_min - η_max)*freq_coeff, 
						  Npt* abs(θ_min - θ_max) * angle_coeff] .|> ceil
  println("TEST : Number of points on the grid : $nb_points_param_grid")
  nb_points_param_grid = convert(Vector{Int64}, nb_points_param_grid)
  # buiding the grid
  g=Array{Array{Float64,1}}(undef,dim);
  a,b=bounds[1],bounds[2];
  for i in 1:dim
    g[i]=collect(range(a[i], stop=b[i], length=nb_points_param_grid[i]));
  end
  #building the meshgrid
  mg1 = ones(length(g[2])) * g[1]'
  mg2 = g[2] * ones(length(g[1]))'
  meshgrid = vcat.(mg1,mg2)
  return spec_lchirp(dim, pt, pω, p, Npt, Npω, Dpt, Dpω, nb_points_param_grid, g, meshgrid, σ, bounds)
end

mutable struct operator_spec_lchirp <: operator
  ker::DataType
  dim::Int64
  sigma::Float64
  bounds::Array{Array{Float64,1},1}

  normObs::Float64

  phi::Function
  d1phi::Function
  d11phi::Function
  d2phi::Function
  y::Array{Float64,1}
  c::Function
  d10c::Function
  d01c::Function
  d11c::Function
  d20c::Function
  d02c::Function
  ob::Function
  d1ob::Function
  d11ob::Function
  d2ob::Function
  correl::Function
  d1correl::Function
  d2correl::Function
end

function setSpecOperator(kernel::spec_lchirp,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1})

	"""phiVect(x)
	Given the parameters x=(ηv, θv), computes the associated spectrogram line.

	ηv and θv are the "visual" arguments, that is those when we consider our observation
	to be on a window of [0,1]x[0,1]. In reality the frequencies are on [0,kernel.Npω-1]
	"""
  function phiVect(x::Array{Float64,1})
	# x is of the form (ηv, θv)
    v=zeros(kernel.Npt*kernel.Npω);
	# index for the loop
    local l=1; 
	local η = x[1]*kernel.Npω
	local θ = atan(tan(x[2]) * kernel.Npω)
	local c = tan(θ)
	local σ = kernel.σ
	
    for j in 1:kernel.Npt
      for i in 1:kernel.Npω
		  ω=kernel.pω[i]
		  t=kernel.pt[j]
		  v[l]= σ * (1 + σ ^ 4 * c ^ 2) ^ (-1//2) * exp(-2 * pi * σ ^ 2 * (ω - η - c * t) ^ 2 / (1 + σ ^ 4 * c ^ 2))
        l+=1;
      end
    end
    return v;
  end

  function d1φη(x::Array{Float64,1})
    v=zeros(kernel.Npt*kernel.Npω);
	# index for the loop
    local l=1; 
	local η = x[1]*kernel.Npω
	local θ = atan(tan(x[2]) * kernel.Npω)
	local c = tan(θ)
	local σ = kernel.σ
	
    for j in 1:kernel.Npt
      for i in 1:kernel.Npω
		ω=kernel.pω[i]
		t=kernel.pt[j]
		v[l]=  4 * σ ^ 3 * (1 + σ ^ 4 * c ^ 2) ^ (-3//2) * pi * (ω - η - c * t) * exp(-2 * pi * σ ^ 2 * (ω - η - c * t) ^ 2 / (1 + σ ^ 4 * c ^ 2))
		v[l] *= kernel.Npω
		l+=1;
      end
    end
    return v;
  end

  function d1φθ(x::Array{Float64,1})
    v=zeros(kernel.Npt*kernel.Npω);
	# index for the loop
    local l=1; 
	local η = x[1]*kernel.Npω
	local θ = atan(tan(x[2]) * kernel.Npω)
	local c = tan(θ)
	local σ = kernel.σ
	
    for j in 1:kernel.Npt
      for i in 1:kernel.Npω
		ω=kernel.pω[i]
		t=kernel.pt[j]
		v[l] = -(c ^ 3 * σ ^ 6 - 4 * t * pi * σ ^ 4 * (η - ω) * c ^ 2 + (-4 * pi * (η - ω) ^ 2 * σ ^ 4 + σ ^ 2 + 4 * pi * t ^ 2) * c + 4 * t * pi * (η - ω)) * (1 + c ^ 2) * σ ^ 3 * exp(-2 * pi * σ ^ 2 * (c * t + η - ω) ^ 2 / (1 + σ ^ 4 * c ^ 2)) * (1 + σ ^ 4 * c ^ 2) ^ (-5//2)
		# ∂θ / ∂θv
		l+=1;
      end
    end
		v .*= kernel.Npω / (cos(x[2])^2 + kernel.Npω^2 * sin(x[2])^2)
    return v;
  end


  function d11φ(x::Array{Float64,1})
    v=zeros(kernel.Npt*kernel.Npω);
	# index for the loop
    local l=1; 
	local η = x[1]*kernel.Npω
	local θ = atan(tan(x[2]) * kernel.Npω)
	local c = tan(θ)
	local σ = kernel.σ
	
    for j in 1:kernel.Npt
      for i in 1:kernel.Npω
		ω=kernel.pω[i]
		t=kernel.pt[j]
		v[l] = 4 * (1 + c ^ 2) * pi * σ ^ 3 * (2 * c ^ 4 * σ ^ 8 * t - 4 * (η - ω) * σ ^ 6 * (pi * t ^ 2 - 3//4 * σ ^ 2) * c ^ 3 + 4 * (-2 * pi * (η - ω) ^ 2 * σ ^ 4 + σ ^ 2 / 4 + pi * t ^ 2) * σ ^ 2 * t * c ^ 2 + 8 * (η - ω) * σ ^ 2 * (-pi * (η - ω) ^ 2 * σ ^ 4 / 2 + 3//8 * σ ^ 2 + pi * t ^ 2) * c + 4 * (-1//4 + pi * (η - ω) ^ 2 * σ ^ 2) * t) * exp(-2 * pi * σ ^ 2 * (c * t + η - ω) ^ 2 / (1 + σ ^ 4 * c ^ 2)) * (1 + σ ^ 4 * c ^ 2) ^ (-7//2)
		v[l] *= kernel.Npω^2 / (cos(x[2])^2 + kernel.Npω * sin(x[2])^2)
		l+=1;
      end
    end
    return v;
  end

  function d1phiVect(m::Int64,x::Array{Float64,1})
    if m==1
		return d1φη(x);
    else
		return d1φθ(x);
    end
  end

  function d11phiVect(i::Int64,j::Int64,x::Array{Float64,1})
	  return d11φ(x);
  end

  function d2φη(x::Array{Float64,1})
    v=zeros(kernel.Npt*kernel.Npω);
	# index for the loop
    local l=1; 
	local η = x[1]*kernel.Npω
	local θ = atan(tan(x[2]) * kernel.Npω)
	local c = tan(θ)
	local σ = kernel.σ
	
    for j in 1:kernel.Npt
      for i in 1:kernel.Npω
		ω=kernel.pω[i]
		t=kernel.pt[j]
		v[l] = 4 * pi * σ ^ 3 * ((4 * pi * σ ^ 2 * t ^ 2 - σ ^ 4) * c ^ 2 + 8 * t * pi * σ ^ 2 * (η - ω) * c - 1 + 4 * pi * (η - ω) ^ 2 * σ ^ 2) * exp(-2 * pi * σ ^ 2 * (c * t + η - ω) ^ 2 / (1 + σ ^ 4 * c ^ 2)) * (1 + σ ^ 4 * c ^ 2) ^ (-5//2)
		v[l] *= kernel.Npω^2
		l+=1;
      end
  end
    return v;
  end

  function d2φθ(x::Array{Float64,1})
    v=zeros(kernel.Npt*kernel.Npω);
	# index for the loop
    local l=1; 
	local η = x[1]*kernel.Npω
	local θ = atan(tan(x[2]) * kernel.Npω)
	local c = tan(θ)
	local σ = kernel.σ
	
	local d1φθtemp = d1φθ(x)
	d1φθtemp /= kernel.Npω / (cos(x[2])^2 + kernel.Npω * sin(x[2])^2)
    for j in 1:kernel.Npt
      for i in 1:kernel.Npω
		ω=kernel.pω[i]
		t=kernel.pt[j]
		v[l] =16 * (cos(θ) * sin(θ) ^ 6 * σ ^ 14 / 8 - pi * σ ^ 12 * t * (cos(θ) ^ 2 + 1) * (η - ω) * sin(θ) ^ 5 / 2 + ((-3//4 - pi * (η - ω) ^ 2 * σ ^ 6 + 3//4 * σ ^ 4 + pi * σ ^ 2 * t ^ 2) * cos(θ) ^ 2 / 2 + σ ^ 2 * (-3//4 * pi * (η - ω) ^ 2 * σ ^ 4 + (-3//16 + t ^ 2 * (η - ω) ^ 2 * pi ^ 2) * σ ^ 2 + 3//4 * pi * t ^ 2)) * cos(θ) * σ ^ 6 * sin(θ) ^ 4 - 2 * pi * (η - ω) * cos(θ) ^ 2 * σ ^ 6 * (σ ^ 2 * cos(θ) ^ 2 / 4 - pi * (η - ω) ^ 2 * σ ^ 4 - 3//4 * σ ^ 2 + pi * t ^ 2) * t * sin(θ) ^ 3 + pi * ((-(η - ω) ^ 2 * σ ^ 6 + t ^ 2 * σ ^ 2) * cos(θ) ^ 2 + pi * ((η - ω) ^ 4 * σ ^ 8 - 4 * t ^ 2 * (η - ω) ^ 2 * σ ^ 4 + t ^ 4)) * cos(θ) ^ 3 * σ ^ 2 * sin(θ) ^ 2 + 2 * pi * (η - ω) * cos(θ) ^ 4 * ((σ ^ 4 - 1) * cos(θ) ^ 2 / 4 + (-pi * (η - ω) ^ 2 * σ ^ 4 + 3//4 * σ ^ 2 + pi * t ^ 2) * σ ^ 2) * t * sin(θ) + ((-pi * (η - ω) ^ 2 * σ ^ 4 + σ ^ 2 / 4 + pi * t ^ 2) * cos(θ) ^ 2 / 2 + 3//4 * pi * (η - ω) ^ 2 * σ ^ 4 + (-3//16 + t ^ 2 * (η - ω) ^ 2 * pi ^ 2) * σ ^ 2 - 3//4 * pi * t ^ 2) * cos(θ) ^ 5) * σ ^ 3 * 0.1e1 / cos(θ) ^ 9 * exp(-2 * pi * σ ^ 2 * (c * t + η - ω) ^ 2 / (1 + σ ^ 4 * c ^ 2)) * (1 + σ ^ 4 * c ^ 2) ^ (-9//2) 
		l+=1;
      end
    end
	v .-= d1φθtemp .* 2(kernel.Npω^2 - 1) * cos(x[2])*sin(x[2]) / kernel.Npω
	v .*= ( kernel.Npω / (cos(x[2])^2 + kernel.Npω * sin(x[2])^2) )^2
    return v;
end

  function d2phiVect(m::Int64,x::Array{Float64,1})
    if m==1
		return d2φη(x);
    else
		return d2φθ(x);
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=dot(phiVect(x1),phiVect(x2));

  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return dot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(2,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return dot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return dot(phiVect(x1),d11phiVect(1,2,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d2phiVect(i,x2));
  end


  y=sum([a0[i]*phiVect(x0[i]) for i in 1:length(x0)])+w;
  normObs=.5*norm(y)^2;

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d11phiVect(k,l,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d2phiVect(k,x),y);
  end

  # TODO: tester si mesh grid fonctionne
  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for pg in kernel.meshgrid
	  PhisY[l]=ob(pg);
	  l +=1;
  end

  function correl(x::Array{Float64,1},Phiu::Array{Array{Float64,1},1})
	  return dot(phiVect(x),sum(Phiu)-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Float64,1},1})
	  d11 = dot(d1phiVect(1,x),sum(Phiu)-y);
	  d12 = dot(d1phiVect(2,x),sum(Phiu)-y);
	  return [d11,d12];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Float64,1},1})
    d2c=zeros(kernel.dim,kernel.dim);
	d2c[1,2]=dot(d11phiVect(0,0,x),sum(Phiu)-y);
	d2c=d2c+d2c';
    d2c[1,1]=dot(d2phiVect(1,x),sum(Phiu)-y);
	d2c[2,2]=dot(d2phiVect(2,x),sum(Phiu)-y);
    return(d2c)
  end

  operator_spec_lchirp(typeof(kernel),kernel.dim,kernel.σ,kernel.bounds,normObs,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function computePhiu(u::Array{Float64,1},op::blasso.operator_spec_lchirp)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiu = [a[i]*op.phi(X[i]) for i in 1:length(a)];
  # Phiux=[a[i]*op.phix(X[i][1]) for i in 1:length(a)];
  # Phiuy=[op.phiy(X[i][2]) for i in 1:length(a)];
  return Phiu;
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Array{Float64,1},1},kernel::blasso.spec_lchirp,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  for pg in kernel.meshgrid
	  buffer = op.correl(pg, Phiu)
      if !positivity
        buffer=-abs(buffer);
      end
      if buffer<correl_min
        correl_min=buffer;
        argmin=pg;
      end
  end

  return argmin,correl_min
end

"""
Sets the amplitude bounds 
"""
function setbounds(op::blasso.operator_spec_lchirp,positivity::Bool=true,ampbounds::Bool=true)
  x_low=op.bounds[1];
  x_up=op.bounds[2];

  if ampbounds
    if positivity
      a_low=0.0;
    else
      a_low=-Inf;
    end
    a_up=Inf;
    return a_low,a_up,x_low,x_up
  else
    return x_low,x_up
  end
end
