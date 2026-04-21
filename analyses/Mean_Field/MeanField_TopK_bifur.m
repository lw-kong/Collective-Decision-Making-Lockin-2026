%% Weighted-TopK(|.|) attention + Degroot-style average (mean-field map)
% Selection compares |omega_o * o| vs |omega_s * m|
% After selection, average uses same weights omega_o, omega_s


%% ---------------- Parameters ----------------
k = 8;                 % number of social inputs (d = k)
G = 1;                 % environment state (+1/-1)
sigma_noise = 1.0/sqrt(0.2);      % o ~ N(G, sigma_noise^2)
step_max = 200;

topK = 3;          % Top-K among (k social + 1 personal), must satisfy 1<=K<=k+1

sw_set = 0:0.02:0.8;
mu0_set = -2.4:0.2:2.4;

result_all = zeros(length(sw_set),length(mu0_set),3);
tic
for sw_i = 1:length(sw_set)
    sw = sw_set(sw_i);

    for mu0_i = 1:length(mu0_set)
        mu0 = mu0_set(mu0_i);
        sigma0 = rand;
        [mu_last,sigma_last,po_last] = func_MeanField_TopK(...
            k,topK,G,sw,sigma_noise,step_max,mu0,sigma0);
        result_all(sw_i,mu0_i,:) = [mu_last,sigma_last,po_last];
    end
    toc
    fprintf('%.3f is done\n',sw_i/length(sw_set))
end

%% plot
plot_mu = zeros(length(sw_set),length(mu0_set));
plot_mu(:,:) = result_all(:,:,1);
plot_po = zeros(length(sw_set),length(mu0_set));
plot_po(:,:) = result_all(:,:,3);

figure('Color','w')
subplot(2,1,1)
scatter(sw_set,plot_mu)
xlabel('Social Weight')
ylabel('Mean Est')
title(['Uniform In-Degree Net, Top ' num2str(topK)])

subplot(2,1,2)
scatter(sw_set,plot_po)
xlabel('Social Weight')
ylabel('Probability of Attention to O')

