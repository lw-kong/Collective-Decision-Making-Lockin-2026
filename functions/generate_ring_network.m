function A = generate_ring_network(N)
% generate_ring_network - 生成一个具有周期边界的环形网络邻接矩阵
%
% 输入:
%   N - 节点数
%
% 输出:
%   A - N×N 的邻接矩阵，表示每个节点与前一个和后一个节点相连（循环）

    A = zeros(N);  % 初始化邻接矩阵

    for i = 1:N
        prev = mod(i-2, N) + 1; % 前一个节点（考虑循环边界）
        next = mod(i, N) + 1;   % 后一个节点（考虑循环边界）

        A(i, prev) = 1;
        A(i, next) = 1;
    end
end
