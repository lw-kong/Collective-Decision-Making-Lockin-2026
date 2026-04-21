function J = jacobian_fd_topK(mu, sg, k, K, omega_s, G, sigma_o, eps_fd)
    x = [mu; max(sg,1e-12)];

    J = zeros(2,2);
    for d = 1:2
        xp = x; xm = x;
        xp(d) = xp(d) + eps_fd;
        xm(d) = xm(d) - eps_fd;

        Fp = func_mf_topKm_step(xp(1), xp(2), k, K, omega_s, G, sigma_o);
        Fm = func_mf_topKm_step(xm(1), xm(2), k, K, omega_s, G, sigma_o);

        J(:,d) = (Fp - Fm) / (2*eps_fd);
    end
end
