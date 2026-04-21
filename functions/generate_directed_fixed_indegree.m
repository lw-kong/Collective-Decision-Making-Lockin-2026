function A = generate_directed_fixed_indegree(N, k)
    % 生成一个有向网络，每个节点的 in-degree 严格等于 k
    % A(i,j) = 1 表示 i -> j

    if k >= N
        error('k must be smaller than N');
    end

    A = zeros(N, N);

    for j = 1:N
        candidates = setdiff(1:N, j);   % 排除自环
        idx = randsample(candidates, k);
        A(idx, j) = 1;
    end
end
