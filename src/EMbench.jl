


module EMbench


using Distributions
using Random
using DataFrames
using BenchmarkTools
using GaussianMixtures   # master branch
using RCall
using StatsPlots  
# deps have https://github.com/floswald/StatsPlots.jl/tree/fo/mix for plotting recipe for Mixtures

function sdata(k,n; doplot = false)
    Random.seed!(3333)
    # true values
    μ = [2.0,5.0]
    σ = [0.5,0.7]
    α = [0.3,0.7]

    m = MixtureModel([Normal(μ[i], σ[i]) for i in 1:2], α)
    if doplot
        plot(
            plot(m,linewidth=2), 
            plot(m,linewidth=2, fill=(0,:red,0.5), components = false, title="Mixture"),dpi = 300
            )
        savefig("mixtures.png")
    end
    y = rand(m,n)

    return Dict(:y => y, :μ => μ, :σ => σ, :α => α)
end

function sdata2(N,T; doplot = false)
    Random.seed!(3333)
    # true values
    μ = [-2.0 10.0;    # μ[k,l]: worker 1 in firm 1 has mean of 2, worker 1 in firm 2 has mean 5
         -3.0 8.0]
    σ = ones(2,2)
    α = [0.3,0.7]
    β = [0.5, 0.5]   # firm shares are given for now!

    m = MixtureModel(vec([Normal(μ[i,j], σ[i,j]) for i in 1:2, j in 1:2]), kron(α,β))
    ps = sampler(m)
    ids = rand(ps.psampler,N,T)
    y   = rand.(m.components[ids])
    firm = copy(ids)
    firm[ids .< 3] .= 1
    firm[ids .> 2] .= 2

    data = DataFrame(i = repeat(1:N, outer = T), t = repeat(1:T, inner = N), y = y[:], l = firm[:])
    @info "true" α=α μ=μ σ=σ

    if doplot
        pl = plot(
            plot(m,linewidth=2), 
            plot(m,linewidth=2, fill=(0,:red,0.5), components = false, title="Mixture"),dpi = 300
            )
        # savefig("mixtures2.png")
        return Dict(:data => data, :μ => μ, :σ => σ, :α => α, :β => β, :plot => pl)
    else
        return Dict(:data => data, :μ => μ, :σ => σ, :α => α, :β => β)
    end

end

function bm_jl2(d::DataFrame; iters=100)

    # poor starting values
    μ = [1.0 4.0;    # μ[k,l]: worker 1 in firm 1 has mean of 2, worker 1 in firm 2 has mean 5
         2.0 7.0]
    σ = ones(2,2)
    α = [0.5,0.5]
    β = [0.5, 0.5]   # firm shares are given for now! irrelevant for this exercise.

    N = maximum(d.i)
    T = maximum(d.t)
    L = maximum(d.l)
    K = length(α)

    # initialize objects    
    Lik = zeros(N,K)  # stationary model: same means and vars in each period
    p = similar(Lik)
    h = zeros(K,L)   # distribution of matches

    for it in 1:iters

        dists = [Normal(μ[ik,il], σ[ik,il] ) for ik in 1:K, il in 1:L]

        fill!(Lik,0.0)

        # evaluate likelihood for each type 
        for i in 1:N
            for k in 1:K
                Lik[i,k] += log(α[k])
                for t in 1:T
                    idx = i + (t-1)*N
                    l = d.l[idx]
                    # Distributions.jl logpdf()
                    Lik[i,k] += logpdf.(dists[k,l], d.y[idx]) 
                end
            end
        end

        # get posterior of each type 
        p[:,:] = exp.(Lik .- logsumexp(Lik))
        # @info "p" p=p
      
        # with p in hand, update type proportions
        α[:] .= vec(sum(p,dims=1) ./ N)

        fill!(μ,0.0)
        fill!(σ,0.0)
        fill!(h,0.0)

        # means need to be updated in a loop 
        for i in 1:N
            for t in 1:T
                idx = i + (t-1)*N
                for k in 1:K
                    pp = p[i,k]  # posterior of worker i to be type k
                    h[k,d.l[idx]] += pp
                    μ[k,d.l[idx]] += pp * d.y[idx]
                end
            end
        end
        μ[:,:] .= μ ./ h


        for i in 1:N
            for t in 1:T
                idx = i + (t-1)*N
                for k in 1:K
                    pp = p[i,k]  # posterior of worker i to be type k
                    σ[k,d.l[idx]] += pp * (d.y[idx] - μ[k,d.l[idx]])^2
                end
            end
        end
        σ[:,:] .= sqrt.(σ ./ h)
        @info "status" α=α μ=μ σ=σ
    end
    return Dict(:α => α, :μ => μ, :σ => σ)
end


function allbm()
    ns = [10_000,100_000,1_000_000,10_000_000]
    d = DataFrame(n = ns,jl = zeros(4),jlg = zeros(4),R = zeros(4),Rmix = zeros(4))
    for ni in ns
        r = bm(25,ni, doplot = ni==ns[1])
        d[d.n .== ni,:jl] = r[:jl]
        d[d.n .== ni,:jlg] = r[:jlg]
        d[d.n .== ni,:R] = r[:R]
        d[d.n .== ni,:Rmix] = r[:Rmix]
    end
    s = stack(d,2:5)
    pl = @df s scatter(:n, :value, group = :variable, 
                        yaxis = :log, xaxis = :log, legend = :topleft,
                        xlabel = "num of observations", 
                        ylabel = "seconds", dpi = 300)
    savefig("EM-bench.png")
    return (d,pl)
end

function bm(iters,N; doplot = false)
    alpha_tol = 0.1  # no convergence, hence big tol
    mu_tol    = 0.5  # no convergence, hence big tol
    tol       = 0.0005  # all algos should converge to same point
    d         = sdata(2,N,doplot = doplot)
    y         = d[:y]
    N         = length(y)
    @info "true values" N=N μ=d[:μ] σ=d[:σ] α=d[:α]

    jl = bm_jl(y,iters = iters)
    # and time
    jl_t = @belapsed bm_jl($y,iters = $iters)
    @info "hand julia: $jl_t seconds"

    jlg = bm_jl_GMM(y,iters = iters)
    jlg_t = @belapsed bm_jl_GMM($y,iters = $iters)
    @info "GMM julia: $jlg_t seconds"

    R = bm_R(y,iters = iters)
    R = rcopy(R)
    @info "hand R: $(R[:time]) seconds"

    Rm = bm_R_mixtools(y,iters = iters)
    Rm = rcopy(Rm)
    @info "R mixtools: $(Rm[:time]) seconds"

    # check outputs

    @assert all( isapprox.( jl[:α], d[:α] , atol = alpha_tol) )
    @assert all( isapprox.( jl[:μ], d[:μ] , atol = mu_tol) )
    @assert all( isapprox.( jl[:σ], d[:σ] , atol = mu_tol) )

    @assert all( isapprox.( jl[:α], jlg.w , atol = tol) )
    @assert all( isapprox.( jl[:μ], jlg.μ , atol = tol) )
    @assert all( isapprox.( jl[:σ], sqrt.(jlg.Σ) , atol = tol) )

    @assert all( jl[:α] .≈ R[:result][:alpha] )
    @assert all( jl[:μ] .≈ R[:result][:mu] )
    @assert all( jl[:σ] .≈ R[:result][:sigma] )

    @assert all( isapprox.( jl[:α] , Rm[:result][:lambda] , atol = tol))
    @assert all( isapprox.( jl[:μ] , Rm[:result][:mu] , atol = tol))
    @assert all( isapprox.( jl[:σ] , Rm[:result][:sigma] , atol = tol))

    return Dict(:jl => jl_t, :jlg => jlg_t, :R => R[:time], :Rmix => Rm[:time])
end


# Naive julia hand implementation

function logsumexp(x::Array{Float64})
    vm = maximum(x,dims = 2)
    log.( sum( exp.( x .- vm ), dims= 2 )) .+ vm
end

function bm_jl(y::Vector{Float64};iters=100)

    # poor starting values
    μ = [4.0,6.0]
    σ = [1.0,1.0]
    α = [0.5,0.5]

    N = length(y)
    K = length(μ)

    # initialize objects    
    L = zeros(N,K)
    p = similar(L)

    for it in 1:iters

        dists = [Normal(μ[ik], σ[ik] ) for ik in 1:K]

        # evaluate likelihood for each type 
        for i in 1:N
            for k in 1:K
                # Distributions.jl logpdf()
                L[i,k] = log(α[k]) + logpdf.(dists[k], y[i]) 
            end
        end

        # get posterior of each type 
        p[:,:] = exp.(L .- logsumexp(L))
      
        # with p in hand, update 
        α[:] .= vec(sum(p,dims=1) ./ N)
        μ[:] .= vec(sum(p .* y, dims = 1) ./ sum(p, dims = 1))
        σ[:] .= vec(sqrt.(sum(p .* (y .- μ').^2, dims = 1) ./ sum(p, dims = 1)))
    end
    return Dict(:α => α, :μ => μ, :σ => σ)
end

function bm_jl_GMM(y::Vector{Float64};iters=100)
    gmm = GMM(2,1)  # initialize an empty GMM object
    # stick in our starting values
    gmm.μ[:,1] .= [4.0;6.0]
    gmm.Σ[:,1] .= [1.0;1.0]
    gmm.w[:,1] .= [0.5;0.5]

    # run em!
    em!(gmm,y[:,:],nIter = iters)
    return gmm
end

function bm_R_mixtools(y::Vector{Float64};iters=100)
    
    r_result = R"""

    library(tictoc)
    library(mixtools)

    mu    = c(4.0,6.0)
    sigma = c(1.0,1.0)
    alpha = c(0.5,0.5)

    y = $y
    N = length(y)
    K = 2
    iters = $iters
    
    tic()
    result = normalmixEM(y,k = K,lambda = alpha, mu = mu, sigma = sigma, maxit = iters)
    tt = toc()
    list(result = result, time = tt$toc - tt$tic)

    """
    return r_result


end


function bm_R(y;iters=100)
    
    r_result = R"""

    library(tictoc)

    # define a `repeat` function
    spread <- function (A, loc, dims) {
        if (!(is.array(A))) {
            A = array(A, dim = c(length(A)))
        }
        adims = dim(A)
        l = length(loc)
        if (max(loc) > length(dim(A)) + l) {
            stop("incorrect dimensions in spread")
        }
        sdim = c(dim(A), dims)
        edim = c()
        oi = 1
        ni = length(dim(A)) + 1
        for (i in c(1:(length(dim(A)) + l))) {
            if (i %in% loc) {
                edim = c(edim, ni)
                ni = ni + 1
            }
            else {
                edim = c(edim, oi)
                oi = oi + 1
            }
        }
        return(aperm(array(A, dim = sdim), edim))
    }

    # define row-wise logsumexp
    logRowSumExp <- function(M) {
        if (is.null(dim(M))) {return(M)}
        vms = apply(M,1,max)
        log(rowSums(exp(M-spread(vms,2,dim(M)[2])))) + vms
    }

    simpleEM <- function(y,iters){
    
        K = 2
        N = length($y)
        
        EMfun <- function(mu,sigma,alpha,iters){
            # allocate arrays
            p = array(0,c(N,K))
            L = array(0,c(N,K))
            
            for (it in 1:iters){
                # E step
                
                # vectorized over N loop
                for (k in 1:K){
                    L[ ,k] = log(alpha[k]) + dnorm(y,mean = mu[k], sd = sigma[k], log = TRUE)
                }
                p = exp(L - logRowSumExp(L))
                
                # M step
                alpha = colMeans(p)
                mu = colSums(p * y) / colSums(p)
                sigma = sqrt( colSums( p * (y - spread(mu,1,N))^2 ) / colSums(p) )
            }
            o =list(alpha=alpha,mu=mu,sigma=sigma)
            return(o)
        }
        
        # starting values
        mu_    = c(4.0,6.0)
        sigma_ = c(1.0,1.0)
        alpha_ = c(0.5,0.5)

        # take time
        tic()
        out = EMfun(mu_,sigma_,alpha_,iters)
        tt = toc()
        return(list(result = out, time = tt$toc - tt$tic))
    }
    simpleEM($y,$iters)


    """
    return r_result


end

end  # module

