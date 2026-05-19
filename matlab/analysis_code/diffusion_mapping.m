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

% Reject zero-sum rows up front: ^-alpha or ^-0.5 of a zero row sum
% silently produces Inf/NaN and corrupts the embedding.
row_sum_data = sum(data, 2);
if any(row_sum_data <= 0)
    error('diffusion_mapping:zeroRowSum', ...
          'Affinity matrix has rows that sum to zero or below.');
end

d = row_sum_data .^ -alpha;

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


% 1. Compute M_diag directly: (data * d) ./ d
% This combines the row sum calculation and the D^2 division
RHS = (data * d) ./ d;

% Check the row sums (see note 1)
if any(RHS <= 0)
    error('diffusion_mapping:zeroRowSum', ...
          'Normalized affinity has rows that sum to zero or below.');
end

RHS = spdiags(RHS, 0, length(RHS), length(RHS)); 
[eigvec_u, eigval] = eigs(data, RHS, length(RHS)); 
eigvec = bsxfun(@rdivide, eigvec_u, d); %Recover v = D^-1 * u
eigval = diag(eigval); 

% Sort eigenvectors and values. eig(Ms,'vector') already returns reals,
% but cast defensively in case of borderline 1e-300 imaginary residue.
[eigval, idx] = sort(real(eigval), 'descend');
eigvec = eigvec(:, idx);

% Scale eigenvectors by the largest eigenvector.
psi = bsxfun(@rdivide, eigvec, eigvec(:,1));

% Automatically determines the diffusion time and scales the eigenvalues.
if diffusion_time == 0
    scaled_eigval = eigval(2:end) ./ (1 - eigval(2:end));
else
    scaled_eigval = eigval(2:end) .^ diffusion_time;
end

% Calculate embedding and bring the data towards output format.
n_available = numel(scaled_eigval);
if n_components > n_available
    warning(['You requested %d components but only %d are available; ' ...
             'returning all available components.'], ...
            n_components, n_available);
    n_components = n_available;
end
embedding = bsxfun(@times, psi(:,2:(n_components+1)), ...
                   scaled_eigval(1:n_components).');
end
