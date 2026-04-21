function [c1, vQ] = quantizer_moments(mu, sigma, Qlevels, taus)
% ===== Helper: 计算多级量化器的 E[Q(e)], Var[Q(e)] =====
% Qlevels: 1xR 升序输出级
% taus:    1x(R+1) 区间阈值，含两端 -inf/+inf，升序
% e ~ N(mu, sigma^2)
    if sigma <= 0
        % 退化：sigma->0 时，全部质量在 mu；直接落到对应区间
        r = find(mu > taus(1:end-1) & mu <= taus(2:end), 1, 'first');
        probs = zeros(size(Qlevels));
        probs(r) = 1;
    else
        z_up   = (taus(2:end)   - mu) / sigma;
        z_down = (taus(1:end-1) - mu) / sigma;
        % 区间概率
        probs = normcdf(z_up, 0, 1) - normcdf(z_down, 0, 1);  % 1xR
        % 数值安全
        probs = max(probs, 0);
        s = sum(probs);
        if abs(s-1) > 1e-12
            probs = probs / s;
        end
    end
    EQ  = sum(Qlevels .* probs);
    EQ2 = sum((Qlevels.^2) .* probs);
    c1  = EQ;
    vQ  = max(EQ2 - EQ.^2, 0);
end
