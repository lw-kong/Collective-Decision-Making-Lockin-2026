function A = generate_dag_ring_network(N, k)
% generate_dag_network - 生成一个无环有向网络邻接矩阵
% 每个节点 i 接受来自右侧紧挨着的 k 个节点 (i+1 ... i+k) 的链接
%
% 输入:
%   N - 节点数
%   k - 每个节点接收的右侧邻居数
%
% 输出:
%   A - N×N 的邻接矩阵，A(i,j)=1 表示 j -> i

    A = zeros(N);  % 初始化邻接矩阵

    for i = 1:N
        % 右侧紧挨的 k 个节点，不能超过 N
        right_neighbors = (i+1) : min(i+k, N);
        A(i, right_neighbors) = 1;
    end
end
