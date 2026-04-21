function [A, xy] = generate_random_geometric_graph(N, rc)
% generate_random_geometric_graph
%   Generate a 2-D random geometric graph in the unit square.
%   N nodes are placed uniformly at random in [0,1] x [0,1].
%   Two nodes are connected if their Euclidean distance is <= rc.
%
% Inputs:
%   N  - number of nodes
%   rc - connection radius
%
% Outputs:
%   A  - NxN logical adjacency matrix (undirected, no self-loops)
%   xy - Nx2 matrix of node coordinates

    if ~isscalar(N) || N <= 0 || round(N) ~= N
        error('N must be a positive integer.');
    end
    if ~isscalar(rc) || rc < 0
        error('rc must be a nonnegative scalar.');
    end

    % Random node positions in the 2-D unit square
    xy = rand(N, 2);

    % Pairwise squared distances
    dx = xy(:,1) - xy(:,1).';
    dy = xy(:,2) - xy(:,2).';
    dist2 = dx.^2 + dy.^2;

    % Build adjacency matrix
    A = dist2 <= rc^2;

    % Remove self-loops
    A(1:N+1:end) = false;

    % Force logical type
    A = logical(A);
end