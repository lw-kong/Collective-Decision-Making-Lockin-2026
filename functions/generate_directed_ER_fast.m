function A = generate_directed_ER_fast(N, k)
    % Calculate the approximate number of edges needed
    % We strictly aim for M edges to approximate average degree k
    M = N * k; 
    
    % 1. Generate random source and target indices
    % We generate a few extra edges (approx 5-10%) to compensate for 
    % collisions (duplicate edges) and self-loops that we will remove.
    %M_oversample = ceil(M * 1.05); 
    M_oversample = M;
    
    ii = randi(N, M_oversample, 1);
    jj = randi(N, M_oversample, 1);
    
    % 2. Create the sparse matrix
    % 'true' creates logical 1s. 
    % MATLAB automatically sums duplicates in sparse(), so we use spones() later
    A = sparse(ii, jj, true, N, N);
    
    % 3. Remove self-loops (diagonal elements)
    % This is very fast on sparse matrices
    if any(diag(A))
        A = spdiags(zeros(N,1), 0, A); 
    end
    
    % 4. Ensure binary (remove weights from duplicate edges)
    A = spones(A);
end