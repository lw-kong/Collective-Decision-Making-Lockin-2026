function A = generate_directed_ER(N, k)
    p = k / (N - 1); % 平均度 -> 连边概率
    A = rand(N, N) < p;  % 每一对节点以概率p连边
    A(1:N+1:end) = 0;    % 去掉自环（对角线元素设为0）
end
