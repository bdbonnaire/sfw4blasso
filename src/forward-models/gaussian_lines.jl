mutable struct gaussianLines <: discrete
  dim::Int64
  px::Array{Float64,1}
  py::Array{Float64,1}
  p::Array{Array{Float64,1},1}
  Npx::Int64
  Npy::Int64
  Dpx::Float64
  Dpy::Float64

  nbpointsgrid::Array{Int64,1}
  grid::Array{Array{Float64,1},1}
  meshgrid::Array{Array{Float64, 1}, 2}

  sigma::Array{Float64,1}
  bounds::Array{Array{Float64,1},1}
end

""" setspecKernel(px, py, Dpx, Dpy, σ, angle_min, angle_max)
Sets the kernel structure `spec_lchirp` corresponding to φ: X -> H in the sfw4blasso paper.

In our case, X=[a_min, a_max]x[θ_min, θ_max] where 
- θ_min, θ_max correspond to `angle_min`, `angle_max` and 
- a_min and a_max are computed from θ_min and θ_max.

# Args
- px, py : Array{Float64,1} 
	the time and frequency grid on H 
- Dpx, Dpy : Float64
	The spacing in `px` and `py`
- sigma : Float64
	window width used to compute the spectrogram
- angle_min, angle_max : Float64
	min and max values for the angle. [angle_min, angle_max] must be included in [-π/2, π/2).
"""
function setGaussLinesKernel(px::Array{Float64,1},py::Array{Float64,1},Dpx::Float64,Dpy::Float64, sigma::Array{Float64,1}, angle_min::Float64, angle_max::Float64)

  dim=2;
  # Sampling of the Kernel
  Npx,Npy=length(px),length(py);
  p=Array{Array{Float64,1}}(undef,Npx*Npy);
  for i in 1:Npy
	  for j in 1:Npx
		  p[(i-1)*Npx+j] = [px[j],py[i]]
	  end
  end

  # Sampling of the parameter space X
  #		TODO: Generalize. This is only verified when considering a signal sampled over 1s. Otherwise the found a might not correlate with frequencies
  ## Computing the bounds of X
  θ_min = angle_min
  θ_max = angle_max
  a_min = -tan(θ_max)
  a_max = 1 - tan(θ_min)
  bounds = [[a_min, θ_min], [a_max, θ_max]]
  println("amin = $a_min et amax = $a_max")
  println("θmin = $θ_min et θmax = $θ_max")
  ## Computing the grid
  freq_coeff = .004
  angle_coeff = .02
  # makes the number of parameter grid samples proportional to the span of [amin, amax] and [θmin, θmax] resp.
  nb_points_param_grid = [ Npy * abs(a_min - a_max)*freq_coeff, 
						  Npx* abs(θ_min - θ_max) * angle_coeff] .|> ceil
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
  return spec_lchirp(dim, px, py, p, Npx, Npy, Dpx, Dpy, nb_points_param_grid, g, meshgrid, sigma, bounds)
end

mutable struct operator_gaussLines <: operator
  ker::DataType
  dim::Int64
  sigma::Array{Float64,1}
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

function setGaussLineOperator(kernel::spec_lchirp,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1})

	"""phiVect(x)
	Given the parameters x=(av, θv), computes the associated spectrogram line.

	av and θv are the "visual" arguments, that is those when we consider our observation
	to be on a window of [0,1]x[0,1]. In reality the frequencies are on [0,kernel.Npy-1]
	"""
  function phiVect(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local a = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		  y=kernel.py[i]
		  x=kernel.px[j]
		  v[l]= √(2) * (π * (σ[1] ^ 2 * sin(θ) ^ 2 + σ[2] ^ 2 * cos(θ) ^ 2)) ^ (-1//2) * exp(-(y - tan(θ) * x - a) ^ 2 / (2 * σ[1] ^ 2 * sin(θ) ^ 2 + 2 * σ[2] ^ 2 * cos(θ) ^ 2)) / 2
        l+=1;
      end
    end
    return v;
  end

  function d1φa(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local a = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		y=kernel.py[i]
		x=kernel.px[j]
		v[l]=  -√(2) * (tan(θ) * x + a - y) * exp(-(tan(θ) * x + a - y) ^ 2 / (2 * σ[1] ^ 2 * sin(θ) ^ 2 + 2 * σ[2] ^ 2 * cos(θ) ^ 2)) * π ^ (-1//2) * (σ[1] ^ 2 * sin(θ) ^ 2 + σ[2] ^ 2 * cos(θ) ^ 2) ^ (-3//2) / 2
		l+=1;
      end
    end
    return v;
  end

  function d1φθ(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local a = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		y=kernel.py[i]
		x=kernel.px[j]
		v[l] = √(2) * ((-2 * σ[1] ^ 2 * σ[2] ^ 2 + σ[2] ^ 4) * sin(θ) * cos(θ) ^ 3 - 2 * x * σ[2] ^ 2 * (y - a) * cos(θ) ^ 2 + (-σ[1] ^ 4 * sin(θ) ^ 2 + (σ[2] ^ 2 + (y - a) ^ 2) * σ[1] ^ 2 + σ[2] ^ 2 * (x + y - a) * (x - y + a)) * sin(θ) * cos(θ) - x * (σ[1] ^ 2 * (x * tan(θ) ^ 3 + 2 * y - 2 * a) * sin(θ) ^ 2 - σ[1] ^ 2 * (y - a) * tan(θ) ^ 2 + 2 * x * tan(θ) * σ[2] ^ 2 - 3 * σ[2] ^ 2 * (y - a))) * exp(-(tan(θ) * x + a - y) ^ 2 / (2 * σ[1] ^ 2 * sin(θ) ^ 2 + 2 * σ[2] ^ 2 * cos(θ) ^ 2)) * π ^ (-1//2) * (σ[1] ^ 2 * sin(θ) ^ 2 + σ[2] ^ 2 * cos(θ) ^ 2) ^ (-5//2) / 2

		l+=1;
      end
    end
    return v;
  end


  function d11φ(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local a = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		y=kernel.py[i]
		x=kernel.px[j]
		v[l] = -exp(-(tan(θ) * x + a - y) ^ 2 / (2 * σ[1] ^ 2 * sin(θ) ^ 2 + 2 * σ[2] ^ 2 * cos(θ) ^ 2)) * √(2) * (σ[1] ^ 2 * sin(θ) ^ 2 + σ[2] ^ 2 * cos(θ) ^ 2) ^ (-7//2) * (-2 * sin(θ) ^ 4 * x * σ[1] ^ 4 - σ[1] ^ 2 * (x * tan(θ) * σ[1] ^ 2 - 3 * (σ[1] ^ 2 - 2//3 * σ[2] ^ 2) * (y - a)) * cos(θ) * sin(θ) ^ 3 - (4 * σ[2] ^ 2 * cos(θ) ^ 2 + x ^ 2 * tan(θ) ^ 4 - x * (y - a) * tan(θ) ^ 3 - σ[1] ^ 2 * tan(θ) ^ 2 + 2 * x * (y - a) * tan(θ) - 3 * σ[2] ^ 2 - 2 * (y - a) ^ 2) * σ[1] ^ 2 * x * sin(θ) ^ 2 + (-2 * σ[2] ^ 2 * (x * (σ[1] ^ 2 - σ[2] ^ 2 / 2) * tan(θ) - 2 * (σ[1] ^ 2 - 3//4 * σ[2] ^ 2) * (y - a)) * cos(θ) ^ 2 + (σ[2] ^ 2 * x ^ 2 - (y - a + σ[1]) * (y - a - σ[1]) * σ[2] ^ 2 + σ[1] ^ 2 * (y - a) ^ 2) * (tan(θ) * x + a - y)) * cos(θ) * sin(θ) - 2 * (cos(θ) ^ 4 * σ[2] ^ 4 + (-σ[1] ^ 2 * tan(θ) ^ 2 / 2 + x * (y - a) * tan(θ) - 3//2 * σ[2] ^ 2 - (y - a) ^ 2) * σ[2] ^ 2 * cos(θ) ^ 2 + (tan(θ) * x + a - y) * (-σ[1] ^ 2 * (y - a) * tan(θ) ^ 2 / 2 + x * tan(θ) * σ[2] ^ 2 - 3//2 * σ[2] ^ 2 * (y - a))) * x) * π ^ (-1//2) / 2
		l+=1;
      end
    end
    return v;
  end

  function d1phiVect(m::Int64,x::Array{Float64,1})
    if m==1
		return d1φa(x);
    else
		return d1φθ(x);
    end
  end

  function d11phiVect(i::Int64,j::Int64,x::Array{Float64,1})
	  return d11φ(x);
  end

  function d2φa(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local a = x[1]
	local θ = x[2]
	local σ = kernel.sigma
	
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		y=kernel.py[i]
		x=kernel.px[j]
		v[l] = exp(-(tan(θ) * x + a - y) ^ 2 / (2 * σ[1] ^ 2 * sin(θ) ^ 2 + 2 * σ[2] ^ 2 * cos(θ) ^ 2)) * √(2) * (tan(θ) ^ 2 * x ^ 2 - 2 * x * (y - a) * tan(θ) - σ[2] ^ 2 * cos(θ) ^ 2 - σ[1] ^ 2 * sin(θ) ^ 2 + (y - a) ^ 2) * (σ[1] ^ 2 * sin(θ) ^ 2 + σ[2] ^ 2 * cos(θ) ^ 2) ^ (-5//2) * π ^ (-1//2) / 2
		l+=1;
      end
  end
    return v;
  end

  function d2φθ(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
	# index for the loop
    local l=1; 
	local a = x[1]*kernel.Npy
	local θ = atan(tan(x[2]) * kernel.Npy)
	local c = tan(θ)
	local σ = kernel.σ
	
	local d1φθtemp = d1φθ(x)
	d1φθtemp /= kernel.Npy / (cos(x[2])^2 + kernel.Npy * sin(x[2])^2)
    for j in 1:kernel.Npx
      for i in 1:kernel.Npy
		y=kernel.py[i]
		x=kernel.px[j]
		v[l] = exp(-(tan(θ) * x + a - y) ^ 2 / (2 * σ[1] ^ 2 * sin(θ) ^ 2 + 2 * σ[2] ^ 2 * cos(θ) ^ 2)) * √(2) * ((4 * x ^ 2 * tan(θ) ^ 2 * σ[1] ^ 6 + σ[1] ^ 8 * cos(θ) ^ 2 + σ[1] ^ 8) * sin(θ) ^ 6 + 8 * x * cos(θ) * σ[1] ^ 6 * (y - a) * sin(θ) ^ 5 + σ[1] ^ 2 * (2 * (2 * σ[2] ^ 2 * σ[1] ^ 4 - 3 * σ[2] ^ 4 * σ[1] ^ 2) * cos(θ) ^ 4 - 12 * σ[1] ^ 2 * (σ[2] ^ 2 * x ^ 2 + σ[1] ^ 2 * (y - a) ^ 2 / 3) * cos(θ) ^ 2 + x ^ 4 * tan(θ) ^ 6 * σ[1] ^ 2 - 3 * x ^ 2 * tan(θ) ^ 4 * σ[1] ^ 4 + 4 * x ^ 3 * σ[1] ^ 2 * (y - a) * tan(θ) ^ 3 - 2 * σ[2] ^ 2 * x ^ 4 * tan(θ) ^ 2 - 6 * x * σ[1] ^ 4 * (y - a) * tan(θ) + 6 * σ[1] ^ 2 * ((31 * x ^ 2 - σ[1] ^ 2) * σ[2] ^ 2 / 6 + (x ^ 2 - σ[1] ^ 2 / 6) * (y - a) ^ 2)) * sin(θ) ^ 4 - 8 * σ[1] ^ 2 * (-3 * cos(θ) ^ 2 * σ[1] ^ 2 * σ[2] ^ 2 + (x ^ 2 + 19//4 * σ[1] ^ 2) * σ[2] ^ 2 + σ[1] ^ 2 * (y - a) ^ 2 / 2) * (y - a) * cos(θ) * x * sin(θ) ^ 3 + (-12 * σ[2] ^ 2 * σ[1] ^ 2 * (σ[2] ^ 2 * x ^ 2 + σ[1] ^ 2 * (y - a) ^ 2) * cos(θ) ^ 4 + ((41 * x ^ 2 * σ[1] ^ 2 + x ^ 4 + (y - a) ^ 4) * σ[2] ^ 4 + 12 * (7//12 * σ[1] ^ 2 + x ^ 2 - (y - a) ^ 2 / 6) * σ[1] ^ 2 * (y - a) ^ 2 * σ[2] ^ 2 + σ[1] ^ 4 * (y - a) ^ 4) * cos(θ) ^ 2 - 2 * x ^ 3 * σ[1] ^ 4 * (y - a) * tan(θ) ^ 5 + 4 * x ^ 4 * tan(θ) ^ 4 * σ[1] ^ 2 * σ[2] ^ 2 + 2 * x * σ[1] ^ 6 * (y - a) * tan(θ) ^ 3 - 6 * σ[1] ^ 4 * (7//3 * σ[2] ^ 2 + (y - a) ^ 2) * x ^ 2 * tan(θ) ^ 2 + 20 * ((x ^ 2 + σ[1] ^ 2 / 2) * σ[2] ^ 2 + σ[1] ^ 2 * (y - a) ^ 2 / 10) * σ[1] ^ 2 * (y - a) * x * tan(θ) - 4 * σ[2] ^ 2 * ((x ^ 4 + 25//4 * x ^ 2 * σ[1] ^ 2) * σ[2] ^ 2 + 6 * σ[1] ^ 2 * (y - a) ^ 2 * (x ^ 2 - σ[1] ^ 2 / 24))) * sin(θ) ^ 2 - 4 * (2 * (-3 * σ[1] ^ 2 * σ[2] ^ 2 + σ[2] ^ 4) * cos(θ) ^ 4 + (-9//2 * σ[2] ^ 4 + (29//2 * σ[1] ^ 2 + (x + y - a) * (x - y + a)) * σ[2] ^ 2 + 2 * σ[1] ^ 2 * (y - a) ^ 2) * cos(θ) ^ 2 + (-13 * σ[1] ^ 2 - 7 * x ^ 2 + 3 * (y - a) ^ 2) * σ[2] ^ 2 / 2 - 3 * σ[1] ^ 2 * (y - a) ^ 2) * σ[2] ^ 2 * (y - a) * cos(θ) * x * sin(θ) + (4 * σ[1] ^ 2 * σ[2] ^ 6 - σ[2] ^ 8) * cos(θ) ^ 8 - 4 * σ[2] ^ 4 * (-σ[2] ^ 4 / 2 + (2 * σ[1] ^ 2 + (x + y - a) * (x - y + a)) * σ[2] ^ 2 + 3 * σ[1] ^ 2 * (y - a) ^ 2) * cos(θ) ^ 6 + 6 * σ[2] ^ 4 * ((σ[1] ^ 2 + 17//3 * x ^ 2 - 5//3 * (y - a) ^ 2) * σ[2] ^ 2 / 2 + (x ^ 2 + 17//6 * σ[1] ^ 2) * (y - a) ^ 2) * cos(θ) ^ 4 - 18 * σ[2] ^ 4 * (7//9 * σ[2] ^ 2 * x ^ 2 + (x ^ 2 + 2//9 * σ[1] ^ 2) * (y - a) ^ 2) * cos(θ) ^ 2 + 4 * x * (x * σ[1] ^ 4 * (y - a) ^ 2 * tan(θ) ^ 4 / 4 - 5//2 * x ^ 2 * σ[1] ^ 2 * σ[2] ^ 2 * (y - a) * tan(θ) ^ 3 + σ[2] ^ 2 * (σ[2] ^ 2 * x ^ 2 + 2 * σ[1] ^ 2 * (y - a) ^ 2) * x * tan(θ) ^ 2 - 3 * σ[2] ^ 2 * (y - a) * (σ[2] ^ 2 * x ^ 2 + σ[1] ^ 2 * (y - a) ^ 2 / 6) * tan(θ) + 13//4 * x * σ[2] ^ 4 * (y - a) ^ 2)) * (σ[1] ^ 2 * sin(θ) ^ 2 + σ[2] ^ 2 * cos(θ) ^ 2) ^ (-9//2) * π ^ (-1//2) / 2
		l+=1;
      end
    end
    return v;
end

  function d2phiVect(m::Int64,x::Array{Float64,1})
    if m==1
		return d2φa(x);
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
