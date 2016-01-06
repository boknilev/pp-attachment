function result = applyNonLinearityDerivative(x)

% result = applyNonLinearity(x) .* (1 - applyNonLinearity(x)); % sigmoid
result = 1 - (applyNonLinearity(x)).^2; % tanh

end