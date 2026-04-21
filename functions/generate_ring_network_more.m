function A = generate_ring_network_more(N, nn)
% generate_ring_network - 生成一个具有周期边界的环形网络邻接矩阵
%
% 输入:
%   N  - 节点数
%   nn - 每个节点连接的两侧最近邻数（每侧 nn 个，共 2*nn 个边）
%
% 输出:
%   A  - N×N 的邻接矩阵

    A = zeros(N);  % 初始化邻接矩阵

    for i = 1:N
        for offset = 1:nn
            left = mod(i - offset - 1, N) + 1;  % 左侧第 offset 个邻居
            right = mod(i + offset - 1, N) + 1; % 右侧第 offset 个邻居
            A(i, left) = 1;
            A(i, right) = 1;
        end
    end
end
