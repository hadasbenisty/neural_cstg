disp(testHyperHiddenDimExtraction(minParams))
function transformedStr = testHyperHiddenDimExtraction(inputStr)
    % Directly match the hyper_hidden_dim pattern
    pattern = 'hyper_hidden_dim\s*=\s*\[([^\]]+)\]';
    match = regexp(inputStr, pattern, 'tokens', 'once');
    
    if ~isempty(match)
        hyper_hidden_dim = match{1}; % Successfully extracted value
    else
        hyper_hidden_dim = 'not found'; % Indicate failure to match
    end
    
    % For demonstration, return the extracted value or 'not found'
    transformedStr = sprintf('Extracted hyper_hidden_dim: %s', hyper_hidden_dim);
end