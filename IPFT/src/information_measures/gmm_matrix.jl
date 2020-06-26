function gmm_matrix(b::AbstractParticleBelief, sigma::Float64)
    N = n_particles(b)
    P = zeros(N, N)
    gmm_matrix!(P, b, sigma)
end

function gmm_matrix!(P::Matrix{Float64}, b::AbstractParticleBelief, sigma::Float64)
    p = particles(b)
    N = length(p)
    P .= 1/(sqrt(2*pi)*sigma) # factor for all entries
    for (i,xi) in enumerate(p), j=i+1:N
        diff = xi - p[j]
        P[i,j] = P[j,i] *= exp(-diff^2 / (2*sigma^2))
    end
    P
end

function gmm_matrix(b::AbstractParticleBelief, sigma::Float64, tol_sigmas::Float64)
    p = particles(b)
    N = length(p)
    P = fill(1/(sqrt(2*pi)*sigma), N, N) # factor for all entries
    for (i,xi) in enumerate(p), j=i+1:N
        diff = xi - p[j]
        if abs(diff) < tol_sigmas
            P[i,j] = P[j,i] *= exp(-diff^2 / (2*sigma^2))
        else
            P[i,j] = P[j,i] = 0.0
        end
    end
    P
end
