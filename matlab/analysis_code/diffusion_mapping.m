function [embedding, scaled_eigval] = diffusion_mapping(data, n_components, alpha, diffusion_time, random_state)
% DIFFUSION_MAPPING   Diffusion mapping decomposition of input matrix.
%   embedding = DIFFUSION_MAPPING(data,n_components,alpha,diffusion_time)
%   computes the first n_components diffusion components of matrix data
%   using parameters alpha and diffusion_time. Variable data must be an
%   n-by-n symmetric matrix containing only real non-negative values,
%   n_components is a natural number, alpha is a scalar in range [0,1], and
%   diffusion_time is a positive scalar. diffusion_time may also be set to
%   0 for automatic diffusion time estimation.
%
%   [embedding, scaled_eigval] = DIFFUSION_MAPPING(data,n_components,alpha, ...
%   diffusion_time) also returns the eigenvalues scaled_eigval.
%
%   For complete documentation please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/diffusion_mapping.html">ReadTheDocs</a>.

if exist('random_state','var')
    rng(random_state);
end

% In future, data can be a function instead of a matrix. This would be useful if
% data is a low rank matrix (eg derived from a 1200 timepoint timeseries where
% there are 32k variables). Then, the large matrix `data` does not have to be
% passed: only a function which can compute data*x. This could be an approach
% which directly makes use of the fact that data is low rank. 
% If this approach is done, replace row_sum_data with something like
% row_sum_data = data * ones(height(data), 1);

% Reject zero-sum rows up front: ^-alpha or ^-0.5 of a zero row sum
% silently produces Inf/NaN and corrupts the embedding.
row_sum_data = sum(data, 2);
if any(row_sum_data <= 0)
    error('diffusion_mapping:zeroRowSum', ...
          'Affinity matrix has rows that sum to zero or below.');
end

d = row_sum_data .^ -alpha;

% ============================================================================ %
% We now do the following calculations:
% D = diag(d); 
% L = D * data * D; % have to check that row sums of L are positive (note 1)
% W = L * ones(width(L), 1); % row-wise degree of L
% solve EVP: W^{-1}Lv = kv (k is eigenvalue and v is eigenvector)

% In a more efficient/stable manner: 
% First, change to generalised EVP:
% Lv = kWv (MATLAB can solve this easily and nicely)
% D * data * D * v = kWv
% data * D * v = kWD^{-1}v (commutes as W and D are diagonal)
% data * u = kWD^{-2}u (where u = Dv, and W and D are nice and diagonal)
% We can even calulate W (diag matrix) without explicitly calculating L!

% note 1: 
% L = D * data * D  -->  L * 1 = d .* (data * d)
% row_sum_L = d .* (data * d);
% has the same sign as d .* (data * d) ./ d.^2 = (data * d) ./ d; 
% ============================================================================ %

% Compute RHS directly: (data * d) ./ d
% This combines the row sum calculation and the D^2 division
RHS = (data * d) ./ d;
if any(RHS <= 0) % Check the row sums (see note 1)
    error('diffusion_mapping:zeroRowSum', ...
          'Normalized affinity has rows that sum to zero or below.');
end
RHS = spdiags(RHS, 0, length(RHS), length(RHS)); 

% Only solve for the number of components that we actually need
n_available = length(d)-1;
if n_components > n_available
    warning(['You requested %d components but only %d are available; ' ...
             'returning all available components.'], ...
            n_components, n_available);
    n_components = n_available;
end

% Main: solve GEVP
% get components with largest algebraic  eigenvalue
% get one more component as first will be discarded 
% configure eigs for optimal performance and stability
% opts.issym = true;  % Forces symmetric solver (use for function input)
opts.isreal = true; % Expect real matrices
[eigvec_u, eigval] = eigs(data, RHS, n_components+1, 'la', opts);
eigvec = bsxfun(@rdivide, eigvec_u, d); % Recover v = D^-1 * u
eigval = diag(eigval);

% Sort eigenvectors and values
[eigval, idx] = sort(eigval, 'descend'); % should error here if eigval is complex
eigvec = eigvec(:, idx);

% Scale eigenvectors by the largest eigenvector.
psi = bsxfun(@rdivide, eigvec(:,2:end), eigvec(:,1));

% Automatically determines the diffusion time and scales the eigenvalues.
if diffusion_time == 0
    scaled_eigval = eigval(2:end) ./ (1 - eigval(2:end));
else
    scaled_eigval = eigval(2:end) .^ diffusion_time;
end

embedding = bsxfun(@times, psi, scaled_eigval.');
end
