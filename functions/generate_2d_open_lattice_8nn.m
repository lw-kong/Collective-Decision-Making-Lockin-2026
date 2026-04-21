function A = generate_2d_open_lattice_8nn(N)
% generate_2d_open_lattice_8nn
%   Generate an LxL 2D lattice adjacency matrix with open boundary conditions
%   and 8-neighborhood (Moore neighborhood): N, S, E, W, and 4 diagonals.
%
% Input:
%   N - number of nodes (must be a perfect square)
%
% Output:
%   A - NxN logical adjacency matrix (undirected, no self-loops)

    L = sqrt(N);
    if abs(L - round(L)) > 1e-12
        error('N must be a perfect square. Got N=%d.', N);
    end
    L = round(L);

    A = false(N, N);

    % Map (r,c) in 1..L to linear index in 1..N (row-major)
    idx = @(r,c) (r-1)*L + c;

    % 8-neighbor offsets (Moore neighborhood)
    nbr_offsets = [ ...
        -1,  0;  % up
         1,  0;  % down
         0, -1;  % left
         0,  1;  % right
        -1, -1;  % up-left
        -1,  1;  % up-right
         1, -1;  % down-left
         1,  1]; % down-right

    for r = 1:L
        for c = 1:L
            i = idx(r,c);

            for k = 1:size(nbr_offsets,1)
                rr = r + nbr_offsets(k,1);
                cc = c + nbr_offsets(k,2);

                % Open boundary: only keep neighbors inside the grid
                if rr >= 1 && rr <= L && cc >= 1 && cc <= L
                    j = idx(rr,cc);

                    if i ~= j
                        A(i,j) = true;
                        A(j,i) = true; % undirected / bidirectional
                    end
                end
            end
        end
    end
end