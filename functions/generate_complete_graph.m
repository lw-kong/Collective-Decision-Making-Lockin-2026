function A = generate_complete_graph(N)
% generate_complete_graph
%   Generate the adjacency matrix of a complete graph with N nodes.
%   Every pair of distinct nodes is connected.
%
% Input:
%   N - number of nodes (positive integer)
%
% Output:
%   A - NxN logical adjacency matrix (undirected, no self-loops)

    if ~isscalar(N) || N <= 0 || round(N) ~= N
        error('N must be a positive integer.');
    end

    % Full connectivity except the diagonal
    A = true(N, N);
    A(1:N+1:end) = false;
    A = logical(A);
end
