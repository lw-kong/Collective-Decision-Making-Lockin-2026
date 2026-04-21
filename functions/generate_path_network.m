function A = generate_path_network(N)
% generate_path_network
%   Generate the adjacency matrix of a path network (chain) with N nodes.
%   End nodes have degree 1, and all internal nodes have degree 2.
%
% Input:
%   N - number of nodes (positive integer)
%
% Output:
%   A - NxN logical adjacency matrix (undirected, no self-loops)

    if ~isscalar(N) || N <= 0 || round(N) ~= N
        error('N must be a positive integer.');
    end

    A = false(N, N);

    % Connect i <-> i+1 for i = 1..N-1
    for i = 1:(N-1)
        A(i, i+1) = true;
        A(i+1, i) = true;
    end
end
