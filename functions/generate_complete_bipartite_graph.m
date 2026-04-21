function A = generate_complete_bipartite_graph(N1, N2)
% generate_complete_bipartite_graph
%   Generate the adjacency matrix of a complete bipartite graph K_{N1,N2}.
%   Nodes are split into two disjoint sets:
%     - Set 1: nodes 1..N1
%     - Set 2: nodes N1+1..N1+N2
%   Every node in Set 1 connects to every node in Set 2.
%   No edges exist within the same set.
%
% Inputs:
%   N1 - number of nodes in partition 1 (positive integer)
%   N2 - number of nodes in partition 2 (positive integer)
%
% Output:
%   A  - (N1+N2)x(N1+N2) logical adjacency matrix (undirected, no self-loops)

    if ~isscalar(N1) || N1 <= 0 || round(N1) ~= N1
        error('N1 must be a positive integer.');
    end
    if ~isscalar(N2) || N2 <= 0 || round(N2) ~= N2
        error('N2 must be a positive integer.');
    end

    N = N1 + N2;
    A = false(N, N);

    % Fill cross-partition edges
    A(1:N1, N1+1:N) = true;
    A(N1+1:N, 1:N1) = true;
end
