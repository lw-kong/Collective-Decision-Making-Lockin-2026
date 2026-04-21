function y = func_discretize_to_list(x, diz_list)
%DISCRETIZE_TO_LIST  Snap scalar/array to nearest value in diz_list
%
%   y = discretize_to_list(x, diz_list)
%
%   INPUTS
%       x         : 任意大小的标量或数值数组
%       diz_list  : 1×N 或 N×1 数值向量，预定义的离散取值集合
%
%   OUTPUT
%       y         : 与 x 尺寸相同的数组，每个元素已替换成 diz_list 中的最近值
%
%   EXAMPLES
%       y1 = discretize_to_list( 0.3 , [-1 1])          % → 1
%       y2 = discretize_to_list([-2.7 2.2], -30:0.5:30) % → [-2.5 2.0]
%
%   NOTE
%       基于隐式扩展 (R2016b+)；若使用老版本，可把减法改成 bsxfun(@minus,…)
%
%   Ling-Wei, 2025-07-15

% 确保 diz_list 是列向量
diz = diz_list(:);          % M×1

% 计算 |x - diz_list| 的矩阵差（隐式扩展）
[~, idx] = min(abs(x(:) - diz.'), [], 2);  % idx: 每个 x 最近值的下标

% 查表并恢复原形状
y = diz(idx);
y = reshape(y, size(x));
end
