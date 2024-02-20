function transformedStr = transformHyperparams(inputStr)
    % Extract parameters using regular expressions with improved pattern matching
    % The patterns now explicitly allow for optional whitespace around '=' and inside brackets
    hyper_hidden_dim_match = regexp(inputStr, 'hyper_hidden_dim\s*=\s*\[([^\]]+)\]', 'tokens', 'once');
    learning_rate_match = regexp(inputStr, 'learning_rate\s*=\s*([0-9.]+)', 'tokens', 'once');
    stg_regularizer_match = regexp(inputStr, 'stg_regularizer\s*=\s*([0-9.]+)', 'tokens', 'once');
    
    % Initialize variables
    hyper_hidden_dim = '';
    learning_rate = '';
    stg_regularizer = '';
    
    % Check if matches were found and assign them
    if ~isempty(hyper_hidden_dim_match)
        hyper_hidden_dim = hyper_hidden_dim_match{1};
    end
    if ~isempty(learning_rate_match)
        learning_rate = learning_rate_match{1};
    end
    if ~isempty(stg_regularizer_match)
        stg_regularizer = stg_regularizer_match{1};
    end
    
    % Format the extracted parameters into the desired string format
    transformedStr = sprintf('c-stg_hidden[%s]_lr%s_lam%s', hyper_hidden_dim, learning_rate, stg_regularizer);
end
