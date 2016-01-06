function result = applyNonLinearity(x)
 
    % result = 1 ./ (1 + exp(-x));  % sigmoid
    result = tanh(x); % tanh

end

function invResult = applyInverseNonLinearity(x)

    % invResult = log(x) - log(1-x); % sigmoid case
    invResult = atanh(x); % tanh case
end